# 05 · 内存访问:stride、二维索引与合并访存

> Learn Triton 系列 · 阶段 1(编程模型)第 2 篇
> 前置:第 02 篇(存储层次)、第 04 篇(二维 grid 与广播索引)
> 运行环境:Google Colab T4 GPU

第 02 篇说过:大多数深度学习算子是 memory-bound,性能 = 你把显存带宽用到几成。而带宽利用率几乎由一件事决定——**访存模式(access pattern)**。本篇讲清楚 Triton 中地址是怎么算出来的(stride)、什么样的访问"合并"(coalesced)、什么样的访问会让带宽掉一个数量级,最后用矩阵转置做一个完整的对比实验。

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

### Tensor 在显存里长什么样:stride(步长)

显存是一维的字节数组。一个 `M×N` 的张量靠 **stride** 把二维坐标翻译成一维偏移:

```text
元素 (i, j) 的地址偏移 = i * stride(0) + j * stride(1)   (单位:元素个数)

行优先(row-major,PyTorch 默认): stride = (N, 1)   同一行的元素相邻
x.t()(转置,零拷贝):            stride = (1, N)   同一行的元素相隔 N!
```

PyTorch 的 `view / transpose / slice` 都只是改 shape 和 stride,**不动数据**。这就是为什么 Triton kernel 总是把 `x.stride(0), x.stride(1)` 作为参数传进去:同一个 kernel 就能处理连续的、转置的、切片的张量。

### 合并访存(memory coalescing):带宽的生死线

GPU 显存以 **32/64/128 字节的事务(transaction)** 为单位读写。第 02 篇讲过 warp 是 32 线程锁步执行——当一个 warp 发起访存时:

- **连续地址**(线程 0 读 `a[0]`、线程 1 读 `a[1]`…):32 个 float32 恰好打包进 1~2 次 128B 事务 → 带宽全吃到;
- **跨步地址**(线程 i 读 `a[i*32]`):每个线程的数据落在不同事务里,硬件被迫发起 32 次事务,**每次 128B 里只有 4B 有用** → 有效带宽掉到 1/32。

Triton 中你不直接控制线程,但你给出的 `offsets` 向量决定了访存模式:**`tl.arange` 落在哪个维度、乘的是哪个 stride,就决定了快慢**。

### 能做什么

- 通过 stride 参数,一个 kernel 通吃连续/转置/切片布局;
- 支持**任意间接寻址(gather/scatter)**:`offsets` 可以来自另一个 `tl.load` 读进来的索引张量——这是 embedding、PagedAttention(第 18 篇)的基础;
- 2D 块 + `tl.trans` 可以在寄存器/片上完成小块转置,把"读和写至少有一边合并"做到极致;
- 编译器会自动把连续访问向量化成宽指令(如 128 位的 `ld.global.v4`)。

### 不能做什么

- **救不了本质上随机的访存**:纯随机 gather 的带宽下限由硬件事务粒度决定,任何编译器都无能为力——能做的是数据布局重排,让随机变成局部(第 18 篇 block 化 KV 就是这个思路);
- Triton 不暴露 shared memory 的显式控制,无法像 CUDA 那样手工设计 bank-conflict-free 的转置缓冲(编译器内部会用 shared memory,但布局你说了不算);
- `tl.load` 的指针必须对齐到元素大小;非对齐的字节级访问不在能力范围内;
- mask 太碎(高度发散)时,即使地址连续,事务利用率也会下降。

---

## §2 递进式例子

### 例 1:一个 kernel 通吃连续与转置布局 —— stride 的威力

```python
@triton.jit
def copy2d_kernel(x_ptr, out_ptr, M, N,
                  sxm, sxn,            # 输入的 stride
                  som, son,            # 输出的 stride
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(x_ptr + rows[:, None] * sxm + cols[None, :] * sxn, mask=mask)
    tl.store(out_ptr + rows[:, None] * som + cols[None, :] * son, x, mask=mask)


def triton_copy(x):
    M, N = x.shape
    out = torch.empty(M, N, device="cuda", dtype=x.dtype)  # 输出总是连续布局
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    copy2d_kernel[grid](x, out, M, N, x.stride(0), x.stride(1),
                        out.stride(0), out.stride(1), BLOCK_M=64, BLOCK_N=64)
    return out


a = torch.randn(1024, 1024, device="cuda")
torch.testing.assert_close(triton_copy(a), a)            # 连续输入
torch.testing.assert_close(triton_copy(a.t()), a.t())    # 转置输入(stride=(1,1024)),同一个 kernel!
print("同一个 kernel 处理连续 与 转置 布局:正确 ✓")
print(f"a 的 stride={tuple(a.stride())}, a.t() 的 stride={tuple(a.t().stride())}(数据没动,只是换了解读方式)")
```

### 例 2:跨步访问有多伤?——stride 扫描实验

固定访问 1600 万个元素,但相邻线程读取的元素间隔从 1 拉大到 32,看有效带宽怎么崩。

```python
@triton.jit
def strided_read_kernel(x_ptr, out_ptr, n, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    # 读取位置:idx * STRIDE —— STRIDE=1 即连续,>1 即跨步
    vals = tl.load(x_ptr + idx * STRIDE, mask=mask)
    tl.store(out_ptr + idx, vals, mask=mask)


n_access = 16 * 1024 * 1024
out = torch.empty(n_access, device="cuda")
print(f"{'STRIDE':>7} | {'耗时 ms':>9} | {'有效带宽 GB/s':>13} | 相对 stride=1")
base = None
for stride in [1, 2, 4, 8, 16, 32]:
    x = torch.randn(n_access * stride, device="cuda")
    ms = gpu_time_ms(lambda: strided_read_kernel[(triton.cdiv(n_access, 1024),)](
        x, out, n_access, STRIDE=stride, BLOCK=1024))
    bw = (2 * n_access * 4) / (ms / 1000) / 1e9  # 有用数据:读+写各 4B
    base = base or bw
    print(f"{stride:>7} | {ms:>9.3f} | {bw:>13.1f} | {bw / base * 100:>6.1f}%")
print("\n(同样'有用'的数据量,跨步越大、浪费的事务越多,带宽断崖式下跌)")
```

### 例 3:间接寻址(gather)—— embedding lookup

索引本身也是从显存读出来的张量。这是"指针的指针",PagedAttention 的 block table(第 18 篇)就是这个模式的工业级应用。

```python
@triton.jit
def embedding_kernel(ids_ptr, table_ptr, out_ptr, n_tokens, dim,
                     BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)                      # 一个 program 负责一个 token
    token_id = tl.load(ids_ptr + pid)           # 第一跳:读出这个 token 的词表行号
    cols = tl.arange(0, BLOCK_D)
    mask = cols < dim
    vec = tl.load(table_ptr + token_id * dim + cols, mask=mask)   # 第二跳:按行号取向量
    tl.store(out_ptr + pid * dim + cols, vec, mask=mask)


vocab, dim, n_tokens = 50_000, 512, 4096
table = torch.randn(vocab, dim, device="cuda")
ids = torch.randint(0, vocab, (n_tokens,), device="cuda", dtype=torch.int64)
out = torch.empty(n_tokens, dim, device="cuda")
embedding_kernel[(n_tokens,)](ids, table, out, n_tokens, dim, BLOCK_D=triton.next_power_of_2(dim))
torch.testing.assert_close(out, table[ids])
print("gather(embedding lookup)正确 ✓ —— 与 torch 的 table[ids] 等价")
print("注意:行内(dim 维)访问仍是连续的,所以 gather 行 ≠ 慢;真正慢的是元素级随机访问")
```

---

## §3 知识连接

**与前面篇章:**

- 第 02 篇的"存储层次":合并访存就是在显存这一层把事务利用率打满;第 04 篇例 1 的 `rows[:, None] * stride_m + cols[None, :] * stride_n` 现在你能解释它的性能含义了——`cols` 乘的 stride 是 1 时,块内最后一维连续,访存合并;
- 第 03 篇说"Triton 默认就快",前提就是你的 offsets 让最后一维连续。写出 `idx * 32` 这种模式,Triton 也救不了你(例 2 实测)。

**与 CUDA 对照:**

- CUDA 教科书里的"全局内存合并访问"规则(warp 内 32 线程访问连续 128 字节)在 Triton 中被翻译为:**让 `tl.arange` 的最快变化维度乘以 stride=1**;
- CUDA 的经典转置优化(shared memory tile + padding 避免 bank conflict)在 Triton 里就是例 1 的 2D 块 + `tl.trans`,shared memory 由编译器代管。

**与真实框架:**

- PyTorch 的 `.contiguous()` 调用的就是一个布局重排 kernel,本篇 `triton_copy(a.t())` 实现的正是 `a.t().contiguous()`;
- vLLM 的 KV cache 写入 kernel(`vllm/attention/ops/` 下的 reshape_and_cache 系列)本质是"gather/scatter + 保持最后一维连续",与例 3 同构;
- Embedding 例子对应 `torch.nn.functional.embedding` 的底层实现思路。

---

## §4 闭环对比实验:矩阵转置的三种写法

转置是访存模式优化的标准考题:读和写**不可能同时**完全连续(读按行连续,写就按列跳;反之亦然)。三种策略对比:

1. **行读列写**:读合并、写不合并;
2. **列读行写**:读不合并、写合并;
3. **2D 分块 + `tl.trans`**:小块整体载入寄存器/片上,转置后整体写出——读写**都以块为单位接近连续**。

对照组:`x.t().contiguous()`(PyTorch 官方实现)。

```python
import matplotlib.pyplot as plt

@triton.jit
def transpose_rowread_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    """策略1:每个 program 读 x 的一行片段(连续),写 out 的一列(跨步)。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    vals = tl.load(x_ptr + pid_m * N + cols, mask=mask)        # 连续读
    tl.store(out_ptr + cols * M + pid_m, vals, mask=mask)      # 跨步写(间隔 M)


@triton.jit
def transpose_colread_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    """策略2:读 x 的一列(跨步),写 out 的一行(连续)。"""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BLOCK + tl.arange(0, BLOCK)
    mask = rows < M
    vals = tl.load(x_ptr + rows * N + pid_n, mask=mask)        # 跨步读(间隔 N)
    tl.store(out_ptr + pid_n * M + rows, vals, mask=mask)      # 连续写


@triton.jit
def transpose_tiled_kernel(x_ptr, out_ptr, M, N,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """策略3:2D tile 整块读入 -> tl.trans 块内转置 -> 整块写出。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tile = tl.load(x_ptr + rows[:, None] * N + cols[None, :],
                   mask=(rows[:, None] < M) & (cols[None, :] < N))      # [BM, BN] 块读,行连续
    tl.store(out_ptr + cols[:, None] * M + rows[None, :], tl.trans(tile),
             mask=(cols[:, None] < N) & (rows[None, :] < M))            # [BN, BM] 块写,行连续


M = N = 8192
x = torch.randn(M, N, device="cuda")
out = torch.empty(N, M, device="cuda")
ref = x.t().contiguous()
bytes_moved = 2 * M * N * 4

candidates = {
    "row-read col-write": lambda: transpose_rowread_kernel[(M, triton.cdiv(N, 1024))](x, out, M, N, BLOCK=1024),
    "col-read row-write": lambda: transpose_colread_kernel[(N, triton.cdiv(M, 1024))](x, out, M, N, BLOCK=1024),
    "2D tiled + tl.trans": lambda: transpose_tiled_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](x, out, M, N, BLOCK_M=64, BLOCK_N=64),
    "PyTorch .t().contiguous()": lambda: x.t().contiguous(),
}

names, bws = [], []
for name, fn in candidates.items():
    fn()
    if "PyTorch" not in name:
        torch.testing.assert_close(out, ref)   # 每种写法先验证正确
    ms = gpu_time_ms(fn)
    bw = bytes_moved / (ms / 1000) / 1e9
    names.append(name); bws.append(bw)
    print(f"{name:28s} {ms:8.3f} ms   {bw:7.1f} GB/s")

plt.figure(figsize=(9, 4))
plt.barh(names, bws)
plt.axvline(320, color="gray", ls="--", label="T4 theoretical 320 GB/s")
plt.xlabel("effective bandwidth (GB/s)")
plt.title(f"Matrix transpose {M}x{N}: access pattern decides everything")
plt.legend(); plt.tight_layout(); plt.show()
```

### 实验结果解读

- 策略 1/2(总有一边跨步)带宽明显低:每个跨步方向的访问把 128B 事务用成 4B,浪费 97%;
- **策略 3(2D 分块)大幅领先**,通常与 PyTorch 官方实现相当:块内"行连续读 + 行连续写",跨步被限制在块与块之间,事务利用率接近满格。这验证了本篇核心论点:**优化 memory-bound 算子 = 重新组织访存模式,而不是优化计算**;
- 面试时能把"为什么转置需要分块"讲到事务粒度这一层,就过关了。

---

## §5 练习 + 面试考点

### 动手练习

1. 把策略 3 的 `BLOCK_M×BLOCK_N` 从 16×16 扫到 128×128,画带宽曲线。太小为什么慢?太大为什么也可能慢?(提示:寄存器压力与 occupancy,第 11 篇正式展开)
2. 改造例 3:让一个 program 处理 4 个 token(内层 `for` 循环),对比 program 数减少后的性能变化。

### 面试高频考点

- **Q:什么是合并访存?不合并会损失多少?**
  A:warp 内 32 线程访问连续地址时,硬件合并为少量 128B 内存事务;跨步/随机访问导致每事务有效字节占比骤降,极端情况带宽掉到 1/32。本篇例 2 的实测:stride=32 时带宽只剩百分之几。
- **Q:PyTorch 里 `transpose` 为什么是零拷贝?什么时候会真正搬数据?**
  A:transpose 只改 stride 元信息;后续遇到需要连续布局的操作(`.contiguous()`、`view` 失败时、部分 kernel 要求)才触发真实搬运。
- **Q:矩阵转置怎么优化?**
  A:2D 分块,块内先读后转再写,使读写都以连续行为单位;CUDA 中用 shared memory tile + padding 防 bank conflict,Triton 中用 2D 块 + `tl.trans`,编译器代管片上缓冲。
- **Q:gather(如 embedding、paged KV)还能合并访存吗?**
  A:行级 gather 可以——只要每行内部连续且足够宽,跨行的随机性影响有限;元素级随机 gather 无法合并,优化手段是改数据布局(把随机粒度变粗,如 PagedAttention 用 16-token 的 block 为单位组织 KV)。
