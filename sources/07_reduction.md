# 07 · 归约(reduction):tl.sum / tl.max 与第一版 softmax

> Learn Triton 系列 · 阶段 1(编程模型)第 4 篇
> 前置:第 04 篇(mask 与 `other` 填充)、第 06 篇(融合)
> 运行环境:Google Colab T4 GPU

逐元素操作(第 06 篇)的每个输出只依赖一个输入;**归约**则相反:一个输出依赖一整片输入——求和、最大值、均值、方差、softmax 的分母,全是归约。它是逐元素与矩阵乘之间的中间难度,也是 LayerNorm(第 12 篇)、FlashAttention(第 14 篇)的核心组件。本篇掌握 Triton 的块内归约原语,并写出第一版完整 softmax。

## 环境准备

```python
import torch

assert torch.cuda.is_available(), "请在 Colab 选择 GPU 运行时"

import triton
import triton.language as tl

print(f"PyTorch {torch.__version__} | Triton {triton.__version__} | {torch.cuda.get_device_name(0)}")


def gpu_time_ms(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]
```

---

## §1 是什么 & 能力边界

### 归约在 Triton 中的形态

Triton 提供**块内归约原语**:对一个已经 load 进来的块(一维或二维)沿某个轴聚合:

```py
s = tl.sum(x, axis=0)      # 求和
m = tl.max(x, axis=0)      # 最大值(还有 tl.min)
# 通用形式:tl.reduce(x, axis, combine_fn) 可自定义结合律操作
```

关键认知:**归约发生在一个 program 内部**。编译器自动生成 warp 内 shuffle、warp 间经 shared memory 的树状归约——这些在 CUDA 里要手写几十行的经典代码,在 Triton 里是一个函数调用。

由此得到 Triton 归约的**基本设计模式**:

- **模式 A(行归约)**:每行一个 program,整行 load 进块里,`tl.sum(axis=0)` 一步出结果。适合"行宽 ≤ 单块容量"(几千~一两万元素),softmax/LayerNorm 都是这个模式;
- **模式 B(分段归约)**:数据太长一个块装不下 → 每个 program 算一段的部分结果,再用第二步(另一个 kernel、torch 收尾、或 atomic)合并。这是"两阶段归约"。

### 能做什么

- 行/列/任意轴的块内归约,一行代码,性能接近手写 CUDA;
- 归约与逐元素操作**自由融合**:softmax = max 归约 + exp 逐元素 + sum 归约 + 除法,全部在一个 kernel、一次显存读写内完成(本篇例 2);
- `tl.reduce` 支持自定义结合律函数,能表达 argmax、welford 方差等非平凡归约;
- 顺序循环 + 累加器可以处理任意长度(第 09 篇 online softmax 就是这种"流式归约")。

### 不能做什么

- **单步跨 program 归约不存在**:第 04 篇讲过 program 间无法同步,所以"全数组求和"必须两阶段(模式 B)或 atomic(无序、浮点不确定性,第 13 篇);
- 块内归约的规模受寄存器/shared memory 限制:BLOCK 太大(如 >16K 元素)会编译失败或性能崩,长行必须换模式;
- 浮点归约的求和顺序与 PyTorch 不同,结果有 1e-6 量级的差异是**正常的**(浮点加法不满足结合律),`assert_close` 要用适当容差;
- 归约轴上的访存如果不连续(比如对列归约而数据行优先),性能照样受第 05 篇合并访存规则的制裁——必要时先转置思路(交换 grid 与块内轴)。

---

## §2 递进式例子

### 例 1:行求和 —— 归约的最小骨架(模式 A)

```python
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)                       # 一行一个 program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)   # sum 的安全填充是 0
    tl.store(out_ptr + row, tl.sum(x, axis=0))


M, N = 4096, 1000
x = torch.randn(M, N, device="cuda")
out = torch.empty(M, device="cuda")
row_sum_kernel[(M,)](x, out, M, N, BLOCK_N=triton.next_power_of_2(N))
torch.testing.assert_close(out, x.sum(dim=1), rtol=1e-5, atol=1e-5)
print("行求和正确 ✓ (注意容差:浮点求和顺序不同,逐位相等是不可能的)")
```

### 例 2:第一版完整 softmax —— 两次归约 + 逐元素,融合在一个 kernel

数值稳定的 softmax 需要三步:`m = max(x)` → `e = exp(x - m)` → `out = e / sum(e)`。PyTorch eager 至少要 3 个 kernel、4 次整行读写;我们一个 kernel、读一次写一次。

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=float("-inf"))  # max 的安全填充是 -inf
    x = x - tl.max(x, axis=0)            # 归约 1:行最大值(数值稳定的关键,第 09 篇详谈)
    num = tl.exp(x)                       # 逐元素:exp(-inf)=0,越界位置自动不贡献
    den = tl.sum(num, axis=0)             # 归约 2:分母
    tl.store(out_ptr + row * stride_m + cols, num / den, mask=mask)


def triton_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_N < 2048 else (8 if BLOCK_N < 8192 else 16)
    softmax_kernel[(M,)](x, out, M, N, x.stride(0), BLOCK_N=BLOCK_N, num_warps=num_warps)
    return out


x = torch.randn(2048, 781, device="cuda")   # 故意用非 2 的幂行宽
torch.testing.assert_close(triton_softmax(x), torch.softmax(x, dim=-1))
print("softmax 正确 ✓ —— 两次归约 + exp + 除法,单 kernel 完成")
```

注意两个 `other` 填充值的不同:max 阶段填 `-inf`(不影响最大值),exp 之后无效位自然变成 `exp(-inf)=0`(不影响求和)——mask 填充值的选择(第 04 篇例 2)在这里第一次真刀真枪地起作用。

### 例 3:全数组求和 —— 两阶段归约(模式 B)

一亿个元素装不进任何单块。标准解法:第一阶段每个 program 输出一个部分和,第二阶段对部分和收尾(部分和已经很短,直接用 torch 或再来一层 kernel)。

```python
@triton.jit
def partial_sum_kernel(x_ptr, partial_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(partial_ptr + pid, tl.sum(x, axis=0))    # 每个 program 写一个部分和


n = 100_000_000
x = torch.randn(n, device="cuda")
BLOCK = 4096
n_blocks = triton.cdiv(n, BLOCK)
partial = torch.empty(n_blocks, device="cuda")

partial_sum_kernel[(n_blocks,)](x, partial, n, BLOCK=BLOCK)
total = partial.sum()                                  # 第二阶段:24415 个数,torch 收尾

torch.testing.assert_close(total, x.sum(), rtol=1e-4, atol=1e-2)
ms = gpu_time_ms(lambda: (partial_sum_kernel[(n_blocks,)](x, partial, n, BLOCK=BLOCK), partial.sum()))
ms_torch = gpu_time_ms(lambda: x.sum())
print(f"两阶段归约: {ms:.3f} ms vs torch.sum: {ms_torch:.3f} ms (1 亿元素)")
print(f"带宽: {n * 4 / (ms / 1000) / 1e9:.0f} GB/s —— 归约是典型 memory-bound,读一遍数据是下限")
```

---

## §3 知识连接

**与前面篇章:**

- 第 04 篇的 `other` 填充值问题在例 2 里成为正确性的命门(max 填 -inf、sum 填 0);
- 第 06 篇的融合思想升级:softmax 把"归约 + 逐元素"融合,流量从 eager 的 ~4 次行读写降到 1 读 1 写——这是融合的第二级形态(第一级:纯逐元素;第三级:matmul epilogue,第 12 篇;终极形态:FlashAttention,第 14 篇);
- 第 02 篇 roofline:softmax 的 AI ≈ 5 FLOP / 8 Byte < 1,妥妥 memory-bound,所以实验部分我们看带宽不看 FLOPS。

**与 CUDA 对照:**

- CUDA 里块内归约要手写:warp shuffle(`__shfl_down_sync`)→ shared memory 跨 warp 合并 → 树状循环,经典面试手写题;Triton 的 `tl.sum` 把这 30 行变成 1 行,生成的机器码就是同一套方案;
- 模式 B 对应 CUDA 的 grid-level reduction(两个 kernel 或 atomic),限制的根源相同:线程块之间无全局 barrier。

**与真实框架:**

- 本篇 softmax 与 Triton 官方教程 02-fused-softmax 同源;PyTorch Inductor 对 `torch.softmax` 生成的 kernel(`triton_per_fused__softmax_*`,per = persistent reduction)结构与例 2 一致;
- vLLM/SGLang 的采样模块里有大量行 softmax/argmax/top-k 归约 kernel;LayerNorm/RMSNorm(第 12 篇)= 本篇模式 A + 仿射变换;
- 例 2 的"单块装不下整行怎么办"这个问题,正是第 09 篇 online softmax 和第 14 篇 FlashAttention 存在的理由——**带着这个问题进入阶段 2**。

---

## §4 闭环对比实验:softmax 三方对决,扫行宽

固定总元素量(约 3200 万),行宽从 256 扫到 16384(行数相应减少),对比 Triton 手写 / PyTorch eager / torch.compile 的有效带宽(softmax 最小流量 = 读 4n + 写 4n 字节)。

```python
import matplotlib.pyplot as plt

eager_softmax = lambda t: torch.softmax(t, dim=-1)
compiled_softmax = torch.compile(eager_softmax)

widths = [256, 512, 1024, 2048, 4096, 8192, 16384]
total = 32 * 1024 * 1024
bw = {"Triton": [], "PyTorch eager": [], "torch.compile": []}

for N in widths:
    M = total // N
    t = torch.randn(M, N, device="cuda")
    torch.testing.assert_close(triton_softmax(t), eager_softmax(t))  # 每个形状先验证
    bytes_moved = 2 * M * N * 4
    for name, fn in [("Triton", triton_softmax), ("PyTorch eager", eager_softmax),
                     ("torch.compile", compiled_softmax)]:
        ms = gpu_time_ms(lambda: fn(t))
        bw[name].append(bytes_moved / (ms / 1000) / 1e9)

print(f"{'行宽':>7} | " + " | ".join(f"{k:>14}" for k in bw))
for i, N in enumerate(widths):
    print(f"{N:>7} | " + " | ".join(f"{bw[k][i]:>11.1f} GB/s"[:17] for k in bw))

plt.figure(figsize=(9, 4.5))
for name, vals in bw.items():
    plt.semilogx(widths, vals, "o-", label=name)
plt.axhline(320, color="gray", ls="--", label="T4 320 GB/s")
plt.xlabel("row width N (total elements fixed at 32M)")
plt.ylabel("effective bandwidth (GB/s)")
plt.title("Row softmax: Triton vs eager vs compile")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 实验结果解读

- **Triton 手写版在常见行宽(512~8192,正好是 Transformer 隐层/词表行宽量级)稳定贴近带宽上限**,常比 eager 快 2~4 倍:eager 的 softmax 虽然内部也有融合,但 PyTorch 通用实现要兼顾各种 dtype/维度组合,调度路径更长;
- torch.compile 大部分形状能追上手写,验证第 06 篇的结论在"规整单行归约"上仍成立;
- **行宽逼近 16K 时,Triton 单块模式开始吃力**(寄存器压力、occupancy 下降)——单块模式 A 的天花板露出来了。怎么办?**分块流式归约**。这正是下一阶段第 09 篇 online softmax 要解决的问题,而它的答案直接通向 FlashAttention。

---

## §5 练习 + 面试考点

### 动手练习

1. 写一个 `row_mean_var_kernel`:一次 load 同时输出每行的均值和方差(提示:`var = E[x²] − E[x]²`,两个 `tl.sum` 共享同一次 load;想想这个公式数值上的隐患,查一下 Welford 算法)。这是第 12 篇 LayerNorm 的直接前置。
2. 把例 3 改成"第二阶段也用 Triton kernel"(对 `partial` 再跑一次 `partial_sum_kernel`),测试与 torch 收尾的性能差异。

### 面试高频考点

- **Q:GPU 上怎么对一亿个数求和?**
  A:两阶段树状归约:第一阶段各线程块算部分和(块内 warp shuffle + shared memory),第二阶段对部分和再归约;或 atomic 累加(快但浮点结果不确定)。单 kernel 内无法全局同步是根本约束。
- **Q:为什么 GPU 浮点求和结果和 CPU/不同实现间不一致?**
  A:浮点加法不满足结合律,并行归约改变求和顺序,误差量级 ~1e-6(fp32)。对策:容差比较、必要时 Kahan 求和或 fp64 累加器。**确定性**要求高的场景(如训练复现)要固定归约顺序,代价是性能。
- **Q:softmax 一共几次显存读写?怎么优化到最少?**
  A:朴素实现 3 个 kernel 至少 4 读 3 写;融合后 1 读 1 写(max、exp、sum 都在片上)。前提是一行装得进一个块;装不下用 online softmax 分块流式更新(下一阶段)。
- **Q:`tl.sum` 在硬件上是怎么执行的?**
  A:编译器生成树状归约:warp 内用 shuffle 指令(寄存器互换,无访存),warp 之间经 shared memory 汇合,log 复杂度。与 CUDA 手写最优实现同构。
