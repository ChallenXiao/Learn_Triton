# 06 · 算子融合(一):逐元素融合 kernel

> Learn Triton 系列 · 阶段 1(编程模型)第 3 篇
> 前置:第 01 篇(kernel 启动开销)、第 02 篇(memory-bound)、第 03 篇(kernel 骨架)
> 运行环境:Google Colab T4 GPU

第 01 篇实验 B 证明了"同样的活,切碎了干更慢";第 02 篇告诉我们逐元素算子全部是 memory-bound。两件事合在一起,推出本系列第一个真正的优化武器——**算子融合(kernel fusion)**:把一串逐元素操作压进一个 kernel,中间结果不落显存。这是 Triton 最常见、收益最稳定的使用场景,也是 vLLM/Liger-Kernel 里数量最多的一类 kernel。

## 环境准备

```python
import torch
import torch.nn.functional as F

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

### 为什么"融合"有效:算一笔字节账

以 `out = relu(x * 2 + y)` 为例(n 个 float32 元素),eager 模式下是 3 个独立 kernel:

```text
kernel 1: t1 = x * 2     读 4n 字节, 写 4n 字节
kernel 2: t2 = t1 + y    读 8n 字节, 写 4n 字节
kernel 3: out = relu(t2) 读 4n 字节, 写 4n 字节
合计:                    读写 28n 字节 + 3 次启动开销 + 2 个临时张量的显存分配
```

融合成 1 个 kernel 后:

```text
fused: out = relu(x*2+y) 读 8n 字节(x,y), 写 4n 字节(out) = 12n 字节
```

**显存流量 28n → 12n,理论加速 2.33 倍**——而且这与计算无关:中间结果 `t1/t2` 全程活在寄存器里。链越长、融合收益越大。这就是第 02 篇"提高算术强度"的具体操作。

### 能做什么

- 任意逐元素操作链(算术、激活、类型转换、dropout、残差相加)都可融合,正确性容易保证(无跨元素依赖);
- 可以融合**多输入多输出**:一个 kernel 同时写出 `out` 和反向需要的中间量(第 13 篇会用到);
- 配合随机数原语 `tl.rand`,dropout 也能融合且**可复现**(seed + offset 决定一切,不需要存 mask);
- 这是 `torch.compile` 自动化程度最高的场景——本篇实验会直接对比手写与自动生成。

### 不能做什么

- **跨元素有依赖就不再是"逐元素融合"**:归约(softmax 的分母)、卷积、矩阵乘需要不同的模式(第 07/10 篇);
- 融合不减少"必须读写的数据":如果链上每个张量本来就只读写一次(比如单个 `relu`),融合没有收益;
- 不能跨越"形状改变"融合:reshape/permute 之后 stride 变了还能处理,但 `matmul → elementwise` 这种要用 epilogue 融合(第 12 篇),不是本篇的纯逐元素模式;
- 收益有上限:流量降到"读必要输入 + 写必要输出"后就到底了,再融合更多操作只省启动开销。

---

## §2 递进式例子

### 例 1:三合一 —— scale + add + relu

```python
@triton.jit
def fused_sar_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    out = tl.maximum(x * 2.0 + y, 0.0)     # 三个算子,零次中间写回
    tl.store(out_ptr + offs, out, mask=mask)


def fused_sar(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    fused_sar_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


x = torch.randn(10_000_000, device="cuda")
y = torch.randn(10_000_000, device="cuda")
torch.testing.assert_close(fused_sar(x, y), torch.relu(x * 2.0 + y))
print("fused scale+add+relu 正确 ✓")
```

### 例 2:GELU —— 超越函数也能融合

GELU(tanh 近似)在 kernel 里就是一段普通数学表达式。注意 `tanh` 用 `sigmoid` 改写(`tanh(z) = 2·sigmoid(2z) − 1`),`tl.sigmoid` 是内置原语。

```python
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # GELU(tanh 近似): 0.5x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    out = 0.5 * x * (1.0 + tanh_inner)
    tl.store(out_ptr + offs, out, mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    n = x.numel()
    gelu_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out


torch.testing.assert_close(triton_gelu(x), F.gelu(x, approximate="tanh"), rtol=1e-4, atol=1e-5)
print("Triton GELU 与 F.gelu(approximate='tanh') 一致 ✓")
```

### 例 3:可复现的融合 dropout —— `tl.rand`

PyTorch 的 dropout 需要生成并(在训练中隐式)保存 mask;Triton 用**计数器型随机数** `tl.rand(seed, offset)`:同样的 seed + 元素位置永远得到同样的随机数,反向传播时**重算**即可,不用存 mask——省一份显存流量。这正是 Triton 官方教程 low-memory-dropout 的思想。

```python
@triton.jit
def fused_add_relu_dropout_kernel(x_ptr, y_ptr, out_ptr, n, p_drop, seed,
                                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = tl.maximum(x + y, 0.0)                      # add + relu
    r = tl.rand(seed, offs)                          # 每个元素一个 [0,1) 随机数,由 (seed, 位置) 决定
    keep = r >= p_drop
    out = tl.where(keep, z / (1.0 - p_drop), 0.0)    # inverted dropout
    tl.store(out_ptr + offs, out, mask=mask)


def fused_ard(x, y, p=0.1, seed=42):
    out = torch.empty_like(x)
    n = x.numel()
    fused_add_relu_dropout_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, p, seed, BLOCK=1024)
    return out


# 验证 1:p=0 时退化为 add+relu,应与 PyTorch 完全一致
torch.testing.assert_close(fused_ard(x, y, p=0.0), torch.relu(x + y))
# 验证 2:同 seed 两次调用结果完全一致(可复现);不同 seed 不同
a1, a2, b = fused_ard(x, y, 0.5, seed=7), fused_ard(x, y, 0.5, seed=7), fused_ard(x, y, 0.5, seed=8)
assert torch.equal(a1, a2) and not torch.equal(a1, b)
# 验证 3:置零比例 ≈ p
zero_ratio = (fused_ard(x, y, 0.3) == 0).float().mean().item()
relu_zero = (torch.relu(x + y) == 0).float().mean().item()  # relu 本身也产生 0
print(f"dropout 后零比例 {zero_ratio:.3f}(relu 自身贡献 {relu_zero:.3f} + p=0.3 的丢弃)✓ 全部验证通过")
```

---

## §3 知识连接

**与前面篇章:**

- 第 01 篇实验 B(启动开销)+ 第 02 篇 roofline(memory-bound)= 本篇的理论依据;§1 的字节账就是 roofline 里"减少 Bytes、提高 AI"的落地;
- 例 3 的 `tl.rand(seed, offs)` 用的偏移正是第 04 篇的 `offsets` 向量——位置即随机性的索引。

**与真实框架:**

- `torch.compile` 的 Inductor 后端对这类模式做**自动融合**,生成的 kernel 名形如 `triton_poi_fused_add_relu_0`(poi = pointwise)——第 28 篇我们把它打印出来与手写版对照;
- vLLM 的 `fused_add_rms_norm`、SGLang/Liger 的 `fused swiglu` 都是"残差相加 + 规范化/激活"的融合,模式与例 1 相同,只是混入了行归约(第 12/16 篇);
- 例 3 的"重算代替存储"思想在更大尺度上就是训练里的 activation checkpointing(第 24 篇)——**用计算换显存**是贯穿全系列的母题。

---

## §4 闭环对比实验:手写融合 vs PyTorch eager vs torch.compile

测试负载:`out = gelu(x * 2 + y)`(4 个逐元素算子:mul、add、以及 gelu 内部的若干)。三个选手:

1. PyTorch eager(多 kernel,多次显存往返);
2. `torch.compile`(自动融合);
3. 手写 Triton 融合 kernel。

```python
import matplotlib.pyplot as plt

@triton.jit
def fused_workload_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x * 2.0 + y
    inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    out = 0.5 * z * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))
    tl.store(out_ptr + offs, out, mask=mask)


def eager_fn(x, y):
    return F.gelu(x * 2.0 + y, approximate="tanh")


def triton_fn(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    fused_workload_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


compiled_fn = torch.compile(eager_fn)  # Inductor 自动融合

n = 32 * 1024 * 1024
x = torch.randn(n, device="cuda")
y = torch.randn(n, device="cuda")

# 正确性:三者一致
torch.testing.assert_close(triton_fn(x, y), eager_fn(x, y), rtol=1e-4, atol=1e-5)
torch.testing.assert_close(compiled_fn(x, y), eager_fn(x, y), rtol=1e-4, atol=1e-5)
print("三种实现结果一致 ✓\n")

ideal_bytes = 3 * n * 4          # 融合后的最小流量:读 x,y 写 out
results = {}
for name, fn in [("PyTorch eager", eager_fn), ("torch.compile", compiled_fn), ("Triton 手写融合", triton_fn)]:
    ms = gpu_time_ms(lambda: fn(x, y))
    bw = ideal_bytes / (ms / 1000) / 1e9   # 以"有效字节/时间"度量,>越接近屋顶越好
    results[name] = ms
    print(f"{name:16s} {ms:8.3f} ms   按最小流量折算 {bw:6.1f} GB/s")

speedup = results["PyTorch eager"] / results["Triton 手写融合"]
print(f"\n手写融合相对 eager 加速: {speedup:.2f}x (理论上限按字节账约 2.3x 量级)")

plt.figure(figsize=(7, 3.5))
plt.barh(list(results.keys())[::-1], list(results.values())[::-1])
plt.xlabel("time (ms), lower is better")
plt.title("gelu(x*2+y), 32M elements, T4")
plt.tight_layout(); plt.show()
```

### 实验结果解读

- **eager 最慢**:gelu(x·2+y) 在 eager 下展开成多个 kernel,中间结果反复进出显存,流量是融合版的 2 倍以上;
- **torch.compile 与手写 Triton 基本打平**:这类规整的 pointwise 链正是编译器的舒适区,Inductor 生成的 kernel 和我们手写的结构几乎一样——**结论:纯逐元素融合,优先用 torch.compile,不必手写**;
- 那为什么还要学手写?因为下一步——**归约(第 07 篇)、matmul epilogue(第 12 篇)、FlashAttention(第 14 篇)、间接寻址(第 18 篇)**——编译器的自动融合能力会逐级失效,而手写 Triton 的价值逐级上升。本篇是分界线的起点。

---

## §5 练习 + 面试考点

### 动手练习

1. 写一个融合的 `bias + tanh + residual` kernel:`out = tanh(x + bias) + x`(bias 是长度为最后一维的向量,需要广播——提示:`offs % N` 取列号)。与 eager 版本对比速度,并算出理论流量比。
2. 给例 3 加一个输出:同时写出 `out` 和 `keep_mask`(int8),对比"存 mask"与"重算 mask"两版的耗时和显存占用。

### 面试高频考点

- **Q:算子融合为什么能加速?加速比怎么估算?**
  A:省 kernel 启动开销 + 省中间结果的显存读写。对 memory-bound 链,加速比 ≈ 融合前总字节数 / 融合后总字节数,可以拿纸笔精确算出(§1 的字节账),实测会略低于理论值。
- **Q:什么样的算子能融合?什么时候融合没用?**
  A:无跨元素依赖的逐元素链最容易;归约/矩阵乘要换模式。当链上每个张量本就只被读写一次、或瓶颈在计算(compute-bound)时,融合收益趋近于零。
- **Q:torch.compile 已经能自动融合了,为什么框架还要手写 Triton kernel?**
  A:自动融合覆盖 pointwise 与简单归约;涉及算法重构(online softmax)、非规则访存(paged KV)、特殊数值处理(量化)、精细的多输出反向设计时,编译器无法自动推导,需要人来写。工程上的顺序是:先 compile,profile 找剩余热点,再手写。
- **Q:dropout 的 mask 一定要存下来吗?**
  A:不一定。用计数器型 RNG(Philox,`tl.rand(seed, offset)`),反向时用同样的 seed+offset 重算 mask,省显存和带宽——典型的"计算换存储"。
