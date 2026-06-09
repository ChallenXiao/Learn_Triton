# 14 · FlashAttention 前向:tiling + online softmax 的合体

> Learn Triton 系列 · 阶段 2(核心算子)第 6 篇(阶段压轴)
> 前置:第 09 篇(online softmax,必须先读懂)、第 10 篇(tl.dot 分块)、第 12 篇(融合阶梯)
> 运行环境:Google Colab T4 GPU

一切铺垫到此收口。FlashAttention 是过去几年大模型系统领域影响力最大的单个 kernel,也是 AI infra 面试**必考题**。它没有发明新数学——它是第 09 篇 online softmax 与第 10 篇 tiling 的精确组合:**attention 的 N×N score 矩阵从头到尾不落显存**。本篇从 IO 账本推起,写出可运行的 Triton 实现,并用"速度 + 峰值显存 + 朴素版 OOM"三重实验闭环。

## 环境准备

```python
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "请在 Colab 选择 GPU 运行时"

import triton
import triton.language as tl
import triton.testing

print(f"PyTorch {torch.__version__} | Triton {triton.__version__} | {torch.cuda.get_device_name(0)}")
```

---

## §1 是什么 & 能力边界

### 朴素 attention 的 IO 灾难

$$O = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V, \qquad Q,K,V \in \mathbb{R}^{N \times d}$$

朴素实现三步走,每步都把 $N \times N$ 矩阵搬进搬出显存:

```text
S = Q @ K^T      写 N² 个数        (N=8192, fp16, 单头: 128MB)
P = softmax(S)   读写各 N²
O = P @ V        读 N²
```

显存**占用** $O(N^2)$,显存**流量** $O(N^2)$ ——而有用的输入输出只有 $O(Nd)$。序列长 8 倍,score 矩阵大 64 倍。计算本身(两个 GEMM)是 Tensor Core 的菜,但全程在等 $N^2$ 数据的搬运:**长序列 attention 是被 IO 杀死的,不是被计算杀死的**。这正是 FlashAttention 论文(Dao et al., 2022)的核心观察:它是 **IO-aware** 算法。

### FlashAttention:score 算完即扔

把 K/V 切成块,沿 KV 维流式扫描。对每个 Q 块维护三个运行量(对照第 09 篇,多了一个 $acc$):

```text
m   : 当前为止 score 的行最大值          (第 09 篇的 m)
l   : 以 m 为基准的指数和               (第 09 篇的 d)
acc : 以 m 为基准的"未归一化输出" Σ p·V

每来一个 KV 块:
  S_blk = Q @ K_blk^T · scale            # 小块 score,只活在寄存器/SRAM
  m_new = max(m, rowmax(S_blk))
  α     = exp(m - m_new)                  # 换基准因子
  P_blk = exp(S_blk - m_new)
  l     = α·l   + rowsum(P_blk)
  acc   = α·acc + P_blk @ V_blk           # 输出累加器同步换基准!
  m     = m_new
最后:O = acc / l
```

与第 09 篇唯一的增量是最后一行的 `acc` 更新:输出是 $\sum_j p_j v_j$,分子的每一项都带着 $e^{-m}$ 基准,所以换基准时 `acc` 也乘 $\alpha$ ——代数上与先算全量 softmax 再乘 V **精确相等**。

**IO 账**:HBM 流量从 $O(N^2)$ 降到 $O(N^2 d / M_{sram})$ 量级(论文记号),直观说:score 矩阵的读写完全消失,Q/K/V/O 各读写 $O(Nd)$ 若干遍。显存占用从 $O(N^2)$ 降到 $O(Nd)$ ——**序列长度的平方墙被推倒**。

### 能做什么

- 任意长序列的精确 attention(数学无近似,误差仅浮点级);
- 显存 $O(N^2) \to O(Nd)$:8K/32K/128K 上下文成为可能,这是当代长上下文 LLM 的硬件前提;
- 长序列下速度大幅领先(本篇实验:数倍);
- 骨架可扩展:causal mask、GQA、滑窗、ALiBi、paged KV……都是在 KV 循环里加几行(第 17/18 篇)。

### 不能做什么

- **短序列收益小甚至为负**:N 小时 score 矩阵本来就小,朴素版的两个 GEMM 反而更规整(实验可见交叉点);
- 反向传播复杂得多:前向不存 P,反向必须**重算**(只存 m/l 与输出),实现/调试成本数倍——本篇只做前向,训练场景直接用官方实现;
- 它优化 IO,不减少 FLOPs:compute-bound 的 prefill 大批量场景,提速封顶在"消掉 IO 等待"的部分;
- 本篇教学版做了简化:no causal、单一 head_dim、fp16、固定块大小——能力边界画在"读懂并能改",生产请用 `F.scaled_dot_product_attention` / flash-attn 库 / vLLM 的 kernel。

---

## §2 递进式例子

### 例 1:朴素实现 + 显存账本

```python
def naive_attention(q, k, v, scale):
    s = (q @ k.transpose(-1, -2)) * scale     # [BH, N, N] 物化!
    p = torch.softmax(s, dim=-1)
    return p @ v


def peak_mem_mb(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


BH, N, D = 8, 4096, 64        # batch*heads=8
q = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q); v = torch.randn_like(q)
scale = D ** -0.5

mem = peak_mem_mb(lambda: naive_attention(q, k, v, scale))
qkv_mb = 3 * BH * N * D * 2 / 1024**2
score_mb = BH * N * N * 2 / 1024**2
print(f"输入 QKV 共 {qkv_mb:.0f} MB;score 矩阵 {score_mb:.0f} MB;实测峰值 {mem:.0f} MB")
print(f"-> N={N} 时中间结果是输入的 {score_mb / qkv_mb:.0f} 倍;N 翻倍它翻 4 倍。这就是要消灭的对象。")
```

### 例 2:FlashAttention 算法的 Python 参考实现 —— 先把数学跑通

```python
def flash_attention_reference(q, k, v, scale, BLOCK_N=512):
    """单头版,逐 KV 块扫描。与 naive 数学等价(精确,无近似)。"""
    N, D = q.shape
    m = torch.full((N,), float("-inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros(N, device=q.device, dtype=torch.float32)
    acc = torch.zeros(N, D, device=q.device, dtype=torch.float32)

    for s0 in range(0, N, BLOCK_N):
        k_blk = k[s0:s0 + BLOCK_N].float()
        v_blk = v[s0:s0 + BLOCK_N].float()
        s_blk = (q.float() @ k_blk.T) * scale            # [N, BLOCK_N] 小块 score
        m_new = torch.maximum(m, s_blk.max(dim=1).values)
        alpha = torch.exp(m - m_new)                      # 换基准
        p_blk = torch.exp(s_blk - m_new[:, None])
        l = alpha * l + p_blk.sum(dim=1)
        acc = alpha[:, None] * acc + p_blk @ v_blk        # 输出累加器同步换基准
        m = m_new
    return (acc / l[:, None]).to(q.dtype)


qs, ks, vs = q[0], k[0], v[0]
ref = naive_attention(qs[None], ks[None], vs[None], scale)[0]
for BN in [128, 512, 1024]:
    out = flash_attention_reference(qs, ks, vs, scale, BLOCK_N=BN)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
print("任意分块下与朴素 attention 一致 ✓ —— FlashAttention 是精确算法")
```

### 例 3:Triton 实现 —— 非 causal 教学版(~60 行)

并行划分:`grid = (Q 块数, batch×head)`。每个 program 拿一个 Q 块,内部循环扫全部 KV 块——例 2 的循环逐行翻译进 kernel。

```python
@triton.jit
def flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_bh, stride_n,           # [BH, N, D] 布局:bh 步长 = N*D, n 步长 = D, d 步长 = 1
    seqlen, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)            # 第几个 Q 块
    pid_bh = tl.program_id(1)           # 第几个 (batch, head)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    base = pid_bh * stride_bh

    # 载入本 program 的 Q 块 [BLOCK_M, D],全程驻留
    q = tl.load(q_ptr + base + offs_m[:, None] * stride_n + offs_d[None, :],
                mask=offs_m[:, None] < seqlen, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in range(0, seqlen, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < seqlen

        k_blk = tl.load(k_ptr + base + offs_n[:, None] * stride_n + offs_d[None, :],
                        mask=kv_mask[:, None], other=0.0)              # [BLOCK_N, D]
        s = tl.dot(q, tl.trans(k_blk)) * sm_scale                       # [BLOCK_M, BLOCK_N]
        s = tl.where(kv_mask[None, :], s, float("-inf"))                # 越界 KV 不参与

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)                                     # 换基准因子
        p = tl.exp(s - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        v_blk = tl.load(v_ptr + base + offs_n[:, None] * stride_n + offs_d[None, :],
                        mask=kv_mask[:, None], other=0.0)               # [BLOCK_N, D]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v_blk)    # 累加器换基准 + P@V
        m_i = m_new

    acc = acc / l_i[:, None]                                            # 最终归一化
    tl.store(o_ptr + base + offs_m[:, None] * stride_n + offs_d[None, :],
             acc.to(tl.float16), mask=offs_m[:, None] < seqlen)


def triton_flash_attention(q, k, v):
    BH, N, D = q.shape
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), BH)
    flash_attn_fwd_kernel[grid](q, k, v, o,
                                q.stride(0), q.stride(1),
                                N, D ** -0.5,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
                                num_warps=4, num_stages=2)
    return o


out = triton_flash_attention(q, k, v)
ref = naive_attention(q, k, v, scale)
torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
print("Triton FlashAttention 前向正确 ✓ (BH=8, N=4096, D=64)")
print("整个 kernel 中从未出现 [N, N] 的张量 —— score 以 64x64 小块生灭于片上")
```

---

## §3 知识连接

**与前面篇章(本篇是阶段 2 的总装):**

- 第 09 篇 online softmax 提供 `m/l` 与换基准;新增的 `acc·α` 一行是第 09 篇练习里"分子也带基准"的自然推论;
- 第 10 篇:两个 `tl.dot`(QK^T 与 PV)就是 GEMM 分块;Q 块常驻、KV 滑窗,正是 tiling 的复用思想;
- 第 12 篇融合阶梯的第四级:**GEMM → softmax → GEMM 三算子合一**,中间张量彻底蒸发——第 12 篇说"跨 GEMM 不能融合是因为跨行归约",FlashAttention 用 online 归约绕开了这堵墙,这正是它的算法贡献;
- 第 02 篇 roofline:优化把 attention 从"被 $N^2$ 流量压在带宽墙上"推向 compute 区。

**与真实框架:**

- `F.scaled_dot_product_attention`:PyTorch 内置调度器,自动在 FlashAttention / memory-efficient / math 三个后端间选择(T4 这种 sm_75 卡不支持官方 flash kernel,会走 memory-efficient 后端——实验里可看到);
- Triton 官方教程 06-fused-attention 是本篇的完整版(causal + 反向);flash-attn 库(Dao 实验室)是 CUDA 生产实现;
- vLLM 的 prefill attention、SGLang 的 extend attention 都是本骨架 + paged KV 间接寻址(第 18 篇);FlashAttention-2 的改进(调换并行循环次序、减少非矩阵乘 FLOPs)与 FA3(Hopper TMA/warp specialization)在第 17 篇梳理。

---

## §4 闭环对比实验:速度 + 峰值显存 + OOM 边界

序列长度 512 → 16384,三个实现:朴素、本篇 Triton、SDPA。各测中位耗时与峰值显存;朴素版在长序列下 OOM 时如实记录——**OOM 本身就是实验结果**。

```python
import matplotlib.pyplot as plt

def bench_one(fn):
    ms = triton.testing.do_bench(fn, return_mode="median")
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return ms, torch.cuda.max_memory_allocated() / 1024**2


seqlens = [512, 1024, 2048, 4096, 8192, 16384]
BH, D = 8, 64
res_t, res_m = {"naive": [], "Triton flash": [], "SDPA": []}, {"naive": [], "Triton flash": [], "SDPA": []}

for N in seqlens:
    qq = torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
    kk, vv = torch.randn_like(qq), torch.randn_like(qq)
    sc = D ** -0.5

    torch.testing.assert_close(triton_flash_attention(qq, kk, vv),
                               F.scaled_dot_product_attention(qq, kk, vv), rtol=1e-2, atol=1e-2)

    impls = {
        "naive": lambda: naive_attention(qq, kk, vv, sc),
        "Triton flash": lambda: triton_flash_attention(qq, kk, vv),
        "SDPA": lambda: F.scaled_dot_product_attention(qq, kk, vv),
    }
    row = f"N={N:>6}: "
    for name, fn in impls.items():
        try:
            ms, mb = bench_one(fn)
            res_t[name].append(ms); res_m[name].append(mb)
            row += f"{name} {ms:8.2f}ms/{mb:7.0f}MB   "
        except torch.cuda.OutOfMemoryError:
            res_t[name].append(float("nan")); res_m[name].append(float("nan"))
            row += f"{name}      OOM!          "
            torch.cuda.empty_cache()
    print(row)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for name in res_t:
    ax1.loglog(seqlens, res_t[name], "o-", label=name)
    ax2.loglog(seqlens, res_m[name], "o-", label=name)
ax1.set_xlabel("seq len N"); ax1.set_ylabel("time (ms)"); ax1.set_title("Latency")
ax2.set_xlabel("seq len N"); ax2.set_ylabel("peak memory (MB)"); ax2.set_title("Peak memory")
for ax in (ax1, ax2):
    ax.grid(True, which="both", alpha=0.3); ax.legend()
plt.suptitle("Attention forward: naive vs Triton flash vs SDPA (T4)")
plt.tight_layout(); plt.show()
```

### 实验结果解读

- **显存曲线**:朴素版按 $N^2$ 飙升并在长序列处 OOM;flash 两家近乎平坦($O(Nd)$)——一张图讲完 FlashAttention 的存在意义;
- **耗时曲线**:短序列(≤1K)三者接近甚至朴素版略优(score 小、GEMM 规整);长序列后朴素版被 $N^2$ 流量拖垮,flash 拉开数倍差距。**交叉点的存在**正是 §1 能力边界的实证;
- 教学版 Triton 与 SDPA(memory-efficient 后端)同量级,说明 60 行已抓住算法全部精髓;生产 kernel 的进一步差距来自 causal 跳块、更深流水线、架构特化(第 17 篇);
- 面试叙事链:**N² 物化 → IO 账 → online softmax 让分块合法 → 累加器换基准 → 显存 O(Nd) + 流量大降**。本篇三个例子就是按这条链组织的,照着讲即可。

---

## §5 练习 + 面试考点

### 动手练习

1. 给例 3 加 **causal mask**:`s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -inf)`,并在循环层面跳过整块越界的 KV(`start_n > (pid_m+1)*BLOCK_M` 时 break 不可用,想想怎么用循环上界实现)。与 `SDPA(is_causal=True)` 对照——这是第 17 篇的预习。
2. 把 BLOCK_M/BLOCK_N 从 32 扫到 128,找 T4 上的最优组合;思考为什么 BLOCK_N 增大时 `tl.exp` 的开销占比会下降(提示:非矩阵乘 FLOPs 与矩阵乘 FLOPs 之比)。

### 面试高频考点

- **Q:FlashAttention 为什么快?它减少了计算量吗?**
  A:没有减少 FLOPs(甚至略增,重算换基准)。快在 IO:朴素版要对 $N^2$ 的 score 矩阵做多轮显存读写,flash 用 tiling+online softmax 让 score 只在片上存在,HBM 流量从 $O(N^2)$ 降到 $O(N^2d/M)$,显存占用降到 $O(Nd)$。它是 IO-aware 算法的代表作。
- **Q:不存 score 矩阵,softmax 的分母怎么办?**
  A:online softmax:维护运行最大值 m 和运行指数和 l,逐块更新;输出累加器同步乘换基准因子 $e^{m_{old}-m_{new}}$,数学恒等无近似。
- **Q:FlashAttention 的反向怎么做?**
  A:前向只存 O、m、l(合成 logsumexp);反向重算每块的 P,再算 dQ/dK/dV。计算多 ~30%,显存省 $O(N^2)$ ——标准的重算换显存。
- **Q:FA1 → FA2 → FA3 的主要改进?**
  A:FA2:重排并行维度(Q 块为外层并行,KV 内层循环),减少非 matmul 操作与跨 warp 通信,GPU 利用率大增;FA3:面向 Hopper,TMA 异步搬运 + warp specialization + FP8 支持。算法核心(tiling+online softmax)三代未变。
- **Q:什么场景 FlashAttention 帮不上忙?**
  A:短序列(IO 本不瓶颈);decode 单 query(M=1,无 Q 块复用,要用 Flash-Decoding 沿 KV 维 split 并行,第 17 篇);以及瓶颈根本不在 attention 的模型(先 profile)。
