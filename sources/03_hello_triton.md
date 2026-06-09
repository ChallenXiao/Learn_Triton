# 03 · Triton 初识:块级编程模型与第一个 kernel

> Learn Triton 系列 · 阶段 0(地基)第 3 篇
> 前置:第 01 篇(异步执行与计时)、第 02 篇(SM/warp/存储层次/roofline)
> 运行环境:Google Colab T4 GPU(Triton 随 PyTorch 2.x 自带,无需单独安装)

前两篇建立了两个事实:① PyTorch eager 模式不会融合算子,小 kernel 启动开销和显存往返是浪费的主要来源;② 性能上限由 roofline 决定。从本篇起,我们拿起解决问题的工具——**Triton**:亲手写 GPU kernel,从一个向量加法开始。

## 环境准备

```python
import sys

import torch

assert torch.cuda.is_available(), (
    "未检测到 GPU!请在 Colab 菜单:代码执行程序 -> 更改运行时类型 -> 选择 T4 GPU"
)

import triton
import triton.language as tl

print(f"PyTorch : {torch.__version__}")
print(f"Triton  : {triton.__version__}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")


def gpu_time_ms(fn, warmup=5, repeat=20):
    """第 01 篇定义的 CUDA Event 计时函数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]
```

---

## §1 是什么 & 能力边界

### Triton 是什么

**Triton 是 OpenAI 开源的 GPU kernel 编程语言和编译器**:你在 Python 里用 `@triton.jit` 装饰一个函数,用 `triton.language`(简写 `tl`)提供的向量化原语描述"**一个数据块(block)上的计算**",编译器把它 JIT 编译成高效的 GPU 机器码(NVIDIA 上经由 PTX)。

它在技术栈中的位置:

```text
PyTorch eager     ----- 易用性最高,算子粒度固定,无法融合
torch.compile     ----- 自动优化,生成的就是 Triton kernel(第 28 篇)
Triton  <== 本系列 ----- 手写 kernel:Python 语法 + 块级抽象,性能接近 CUDA
CUDA C++          ----- 完全控制每个线程,开发成本最高
PTX / SASS        ----- 汇编层
```

### 核心思想:块级(block-level)编程

这是 Triton 与 CUDA 最本质的区别,也是它易学的原因:

| | CUDA | Triton |
|---|------|--------|
| 你编程的对象 | **单个线程**(再乘以几万个) | **一个数据块**(program) |
| 索引计算 | `blockIdx.x * blockDim.x + threadIdx.x`,逐线程 | `tl.arange(0, BLOCK)` 一次拿到整个块的索引向量 |
| 线程如何分工 | 你手动安排 | **编译器自动安排**(warp 划分、向量化访存) |
| shared memory | 手动声明、手动同步(`__syncthreads`) | 编译器自动管理 |
| 心智模型 | 几万个工人各拿一张工单 | 你给每个车间发一批货的加工说明 |

写 Triton 时你思考的单位是:"第 `pid` 个程序实例负责第 `pid` 块数据,对这块数据做向量化的 load → 计算 → store"。块内 32 线程一组的 warp 调度、访存合并(第 02 篇)等脏活,编译器替你完成。

### 能做什么

- 写出**性能接近甚至超过手写 CUDA** 的算子(尤其是融合类算子),而代码量只有 CUDA 的几分之一;
- 与 PyTorch 无缝互操作:输入输出都是 `torch.Tensor`,可封装进 `nn.Module` 和 autograd(第 13 篇);
- 一份代码跨 NVIDIA / AMD GPU 运行(编译后端不同);
- 是 vLLM、SGLang、Liger-Kernel、torch.compile/Inductor 等主流项目自定义算子的实现语言。

### 不能做什么(能力边界,面试常考)

- **不能在没有 GPU 的机器上运行**(macOS 本地不行;CPU 解释模式 `TRITON_INTERPRET=1` 仅用于调试,极慢);
- **不适合线程级的精细控制**:warp 级原语、精巧的 shared memory 布局(如手写 bank-conflict-free 转置)、warp specialization 等,CUDA/CUTLASS 仍是上限更高的选择——这也是 FlashAttention-3 官方实现回到 CUDA 的原因之一;
- **kernel 内不能调用 PyTorch / NumPy / 任意 Python 库**:`@triton.jit` 函数体只能用 `tl.*` 原语和简单 Python 控制流,它是被编译的 DSL,不是普通 Python;
- **不负责 kernel 之间的调度**:多 kernel 的流水线、多 GPU 通信、CPU 逻辑,都在它的边界之外(分别由框架调度器、NCCL 等负责,见阶段 3/4);
- 块形状必须是**编译期常量**(`tl.constexpr`)且通常要求 2 的幂;动态形状靠 mask 处理。

---

## §2 递进式例子

### 例 1:最小可运行的 Triton kernel —— 向量加法

逐行读懂这 12 行,你就理解了 Triton 的全部骨架:**grid 划分 → program_id 定位 → 算偏移 → mask 防越界 → load/算/store**。

```python
@triton.jit
def add_kernel(
    x_ptr,                      # 输入张量 x 的首地址(指针)
    y_ptr,                      # 输入张量 y 的首地址
    out_ptr,                    # 输出张量的首地址
    n_elements,                 # 元素总数(运行期变量)
    BLOCK_SIZE: tl.constexpr,   # 每个 program 处理的元素数(编译期常量)
):
    # 1) 我是第几个 program?(对应 grid 的第 0 维)
    pid = tl.program_id(axis=0)
    # 2) 本 program 负责的元素下标:[pid*B, pid*B+1, ..., pid*B+B-1]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 3) 末尾的 program 可能越界,mask 标记哪些下标有效
    mask = offsets < n_elements
    # 4) 把一整块数据从显存读进来(向量化 load,mask 处不读)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # 5) 块上的逐元素计算(发生在寄存器里)
    out = x + y
    # 6) 写回显存
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    out = torch.empty_like(x)
    n = out.numel()
    # grid:启动多少个 program。用 lambda 是为了让 grid 能依赖编译期参数
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


# 正确性验证:与 PyTorch 原生结果逐元素比对
x = torch.randn(98_432, device="cuda")  # 故意取一个不能被 1024 整除的长度,检验 mask
y = torch.randn(98_432, device="cuda")
torch.testing.assert_close(triton_add(x, y), x + y)
print("正确性验证通过:triton_add == torch.add")
```

### 例 2:grid 是怎么回事——亲眼看看每个 program 拿到了什么

用一个"调试 kernel"把每个 program 的 `pid` 和处理范围写出来,建立 grid → 数据块的直观映射。

```python
@triton.jit
def whoami_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # 每个元素记下"我是被几号 program 处理的"
    tl.store(out_ptr + offsets, pid, mask=mask)


n = 20
out = torch.full((n,), -1, device="cuda", dtype=torch.int32)
BLOCK = 8
grid = (triton.cdiv(n, BLOCK),)  # = (3,)
whoami_kernel[grid](out, n, BLOCK_SIZE=BLOCK)
print(f"n={n}, BLOCK_SIZE={BLOCK}, grid={grid}")
print("每个元素由哪个 program 处理:", out.cpu().tolist())
print("-> program 0 管 [0,8), program 1 管 [8,16), program 2 管 [16,20)(靠 mask 截断)")
```

### 例 3:看一眼编译产物——Python 到 PTX

Triton 是真编译器,不是封装库。下面取出刚才那个 kernel 编译出的 PTX(NVIDIA 虚拟汇编),不需要看懂,只需确认:你写的 Python 真的变成了机器指令。

```python
# kernel 启动后,编译缓存里能取到编译产物(不同 Triton 版本接口略有差异,做了兼容处理)
handle = None
try:
    cache = list(add_kernel.cache[0].values()) if hasattr(add_kernel, "cache") else []
    handle = cache[0] if cache else None
except Exception:
    pass

if handle is not None and hasattr(handle, "asm"):
    print("编译产物包含:", list(handle.asm.keys()))
    ptx = handle.asm.get("ptx", "")
    print("--- PTX 前 25 行 ---")
    print("\n".join(ptx.splitlines()[:25]))
else:
    print("当前 Triton 版本未暴露 cache 接口,可改用环境变量 MLIR 调试:")
    print("  TRITON_KERNEL_DUMP=1 python your_script.py")
```

### 例 4:同一个 kernel,顺手就能换成别的逐元素操作

体会 Triton 的开发效率:把第 6 行的 `x + y` 换成任何逐元素表达式,就得到一个新算子——这为第 06 篇的算子融合埋下伏笔。

```python
@triton.jit
def fused_mul_add_relu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = tl.maximum(x * 2.0 + y, 0.0)  # relu(x*2 + y):三个算子,一次显存往返
    tl.store(out_ptr + offsets, out, mask=mask)


out = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
fused_mul_add_relu_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=1024)
torch.testing.assert_close(out, torch.relu(x * 2.0 + y))
print("融合 kernel 正确性验证通过:一个 kernel = relu(x*2+y) 三个 PyTorch 算子")
```

---

## §3 知识连接

**与第 01/02 篇的联系:**

- 第 01 篇实验 B 证明了 kernel 启动开销的代价,例 4 给出解法:把多个操作写进**一个** kernel;
- 第 02 篇的 SIMT/warp:`add_kernel` 里 `BLOCK_SIZE=1024` 的一个 program,会被编译器拆成 1024/32 = 32 个 warp 调度执行(可通过 `num_warps` 参数干预,第 11 篇);
- 第 02 篇的访存合并:`tl.arange` 产生连续下标,编译器自动生成合并访存指令——这是 Triton"默认就快"的关键。

**与 CUDA 的对照(有 CUDA 背景的读者):**

- `pid = tl.program_id(0)` ≈ `blockIdx.x`;一个 program ≈ 一个 thread block;
- `offsets = pid*B + tl.arange(0,B)` 把 CUDA 里"每个线程算一个全局下标"折叠成一行向量化表达;
- 没有 `threadIdx`、没有 `__syncthreads`:线程级细节整体下沉给编译器。

**与真实框架的联系:**

- `torch.compile` 对逐元素融合的处理方式与例 4 同构:Inductor 后端自动生成形如 `triton_poi_fused_*` 的 kernel(第 28 篇我们去读它);
- vLLM 仓库 `vllm/attention/ops/` 与 SGLang 的 `sgl-kernel` 中有大量 `@triton.jit` kernel,骨架与本篇完全一致——学完本系列你可以直接读懂;
- Triton 官方教程 01-vector-add(triton-lang.org/main/getting-started/tutorials/)是本篇例 1 的出处,本系列与官方教程可互为参照。

---

## §4 闭环对比实验:Triton vs PyTorch 的向量加法

向量加法是 memory-bound 算子(第 02 篇:AI≈1/12),所以**比较标准不是 FLOPS,而是有效带宽**。我们扫不同规模,对比三件事:Triton kernel、`torch.add`、以及"理论带宽天花板"。

```python
import matplotlib.pyplot as plt

sizes = [2**i for i in range(12, 27, 2)]  # 4K ~ 64M 元素
bw_triton, bw_torch = [], []

for n in sizes:
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    bytes_moved = 3 * n * 4  # 读 a、读 b、写 out,各 4 字节

    ms_t = gpu_time_ms(lambda: triton_add(a, b))
    ms_p = gpu_time_ms(lambda: a + b)
    bw_triton.append(bytes_moved / (ms_t / 1000) / 1e9)
    bw_torch.append(bytes_moved / (ms_p / 1000) / 1e9)

print(f"{'元素数':>12} | {'Triton GB/s':>12} | {'PyTorch GB/s':>12}")
for n, bt, bp in zip(sizes, bw_triton, bw_torch):
    print(f"{n:>12,} | {bt:>12.1f} | {bp:>12.1f}")

plt.figure(figsize=(9, 4.5))
plt.semilogx(sizes, bw_triton, "o-", label="Triton add_kernel")
plt.semilogx(sizes, bw_torch, "s-", label="torch.add")
plt.axhline(320, color="gray", ls="--", label="T4 theoretical 320 GB/s")
plt.xlabel("number of elements")
plt.ylabel("effective bandwidth (GB/s)")
plt.title("Vector add: Triton vs PyTorch (memory-bound)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 实验结果解读

- 大规模时,**Triton 与 PyTorch 的曲线几乎重合**,都贴近实测带宽上限(理论值的 75%~85%)——这正是预期结果:memory-bound 算子的性能由字节数决定,两边读写的字节一样多,谁也不可能更快。**我们用 12 行 Python 写出了与 PyTorch 内置 C++/CUDA 算子同速的 kernel**,这就是 Triton 的价值证明;
- 小规模时两条曲线都远离屋顶:kernel 启动开销主导(第 01 篇实验 B 的重演),且任务太小填不满 40 个 SM;
- 单算子做到同速只是起点;Triton 真正的收益在 PyTorch 做不到的事——**融合**(例 4 已预演,第 06 篇定量测量)。

---

## §5 练习 + 面试考点

### 动手练习

1. 写一个 `triton_scale_shift(x, alpha, beta) = alpha * x + beta` kernel(标量参数直接作为 kernel 入参传入),用 `assert_close` 对照 PyTorch 验证,并测它的有效带宽。提示:这次只读一个张量,`bytes = 2 * n * 4`。
2. 把例 1 的 `BLOCK_SIZE` 改成 128 / 256 / 4096 重跑实验,观察带宽变化并尝试解释(下一篇第 04 篇将系统回答这个问题)。

### 面试高频考点

- **Q:Triton 和 CUDA 的本质区别是什么?**
  A:编程粒度不同。CUDA 是线程级编程,程序员描述单线程行为并手动管理 shared memory 与同步;Triton 是块级编程,程序员描述一个数据块上的向量化计算,线程划分、访存合并、shared memory 分配由编译器完成。代价是放弃线程级精细控制,换来开发效率和"默认就快"。
- **Q:Triton kernel 里的 mask 是干什么的?不写会怎样?**
  A:数据长度通常不被 BLOCK_SIZE 整除,最后一个 program 的部分下标会越界;mask 让 load/store 跳过无效位置(load 可用 `other=` 给默认值)。不写则越界访存,轻则读到垃圾数据、重则 illegal memory access。
- **Q:为什么 BLOCK_SIZE 必须是 `tl.constexpr`?**
  A:它决定寄存器分配、shared memory 用量和指令生成,必须在编译期已知;每个不同的 constexpr 取值组合会触发一次独立编译并缓存(这也是 autotune 的工作机制,第 11 篇)。
- **Q:什么时候应该手写 Triton,而不是用 torch.compile?**
  A:torch.compile 擅长自动融合规整的逐元素/归约模式;涉及非平凡数据流的算法级优化(FlashAttention 的 online softmax、PagedAttention 的间接寻址、量化 GEMM 的特殊布局)编译器推导不出来,需要手写。判断标准:先 profile,确认瓶颈 kernel 且编译器搞不定,再动手。
