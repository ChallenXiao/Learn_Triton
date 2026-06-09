# 09 · 数值稳定 softmax 与 online softmax:FlashAttention 的数学地基

> Learn Triton 系列 · 阶段 2(核心算子)第 1 篇
> 前置:第 07 篇(单块行 softmax 及其"装不下长行"的遗留问题)、第 08 篇(do_bench)
> 运行环境:Google Colab T4 GPU

第 07 篇的 softmax 有个硬限制:**一行必须装进一个块**。行宽 10 万(长序列 attention 的 score 行)怎么办?本篇引入 2018 年 NVIDIA 提出的 **online softmax** 算法:把"求 max"和"求 sum"合并成一次流式扫描,块与块之间只传递两个标量。这个看似小众的数值技巧,正是 FlashAttention(第 14 篇)能把 attention 矩阵"边算边扔"的全部数学依据——**面试中讲不清 online softmax,就讲不清 FlashAttention**。

## 环境准备

```python
import torch

assert torch.cuda.is_available(), "请在 Colab 选择 GPU 运行时"

import triton
import triton.language as tl
import triton.testing

print(f"PyTorch {torch.__version__} | Triton {triton.__version__} | {torch.cuda.get_device_name(0)}")
```

---

## §1 是什么 & 能力边界

### 问题一:为什么 softmax 必须减最大值

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

fp32 的 `exp` 在 $x > 88.7$ 时上溢为 `inf`(fp16 在 $x > 11.1$!),`inf/inf = NaN`。利用恒等式

$$\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j$$

减去行最大值后指数全部 ≤ 0,$e^{x-m} \in (0, 1]$,永不上溢。这就是 **safe softmax**,代价是要先扫一遍求 $m$:**3 遍数据(max → exp&sum → divide)**。

### 问题二:online softmax —— 把 max 和 sum 合成一遍扫描

把一行切成块,顺序处理。维护两个**运行статистики**:当前为止的最大值 $m$ 和"以当前 $m$ 为基准的指数和" $d$。来了新块 $x^{(k)}$:

$$m_{\text{new}} = \max(m_{\text{old}},\ \max(x^{(k)}))$$
$$d_{\text{new}} = d_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum_i e^{x^{(k)}_i - m_{\text{new}}}$$

关键在第二个式子:**旧的和用 $e^{m_{old}-m_{new}}$ 一乘就完成了"换基准"**(因为 $\sum e^{x-m_{old}} \cdot e^{m_{old}-m_{new}} = \sum e^{x-m_{new}}$)。扫描结束时 $(m, d)$ 与全量计算完全一致——**精确算法,不是近似**。数据遍数:3 → 2(一遍求统计量,一遍归一化写出)。

更进一步:如果把"输出也带着累加"(attention 里的 $\sum p_i v_i$),连第二遍都能省——那就是 FlashAttention,第 14 篇见。

### 能做什么

- 处理**任意长度的行**:块与块之间只传 2 个标量,行宽不再受单块容量限制;
- 数学上**精确**等价于 safe softmax(不引入近似误差);
- 统计量更新满足结合律,块的处理顺序可并行重排(这是 Flash-Decoding split-K 的依据,第 17 篇);
- 同样的"换基准"思想适用于一切 log-sum-exp 型计算(logsumexp、交叉熵、第 23 篇 fused CE)。

### 不能做什么

- **不减少计算量**:exp 次数不变甚至略增(多了换基准的乘法),收益全部在**减少显存遍数**——所以只对 memory-bound 场景有意义(softmax 恰好是);
- 不解决 softmax 本身的其他数值问题(如下溢出到 0 的精度损失、fp16 中 $d$ 的精度——实践中 $m,d$ 用 fp32 累加器,即使输入是 fp16);
- 流式更新引入顺序依赖,单行内部的块只能串行(并行化要靠"分段算 + 段间合并",见练习 2)。

---

## §2 递进式例子

### 例 1:亲眼看 naive softmax 爆炸

```python
def naive_softmax(x):
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)


def safe_softmax(x):
    x = x - x.max(dim=-1, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)


x_normal = torch.randn(4, 8, device="cuda")
x_large = x_normal + 100.0   # attention score 在长序列/大 logit 下完全可能到这个量级

print("正常输入: naive 与 torch 一致?",
      torch.allclose(naive_softmax(x_normal), torch.softmax(x_normal, -1)))
print("大数输入: naive 输出 ->", naive_softmax(x_large)[0, :4].tolist(), "  全是 NaN!")
print("大数输入: safe  输出 ->", [round(v, 4) for v in safe_softmax(x_large)[0, :4].tolist()], " 正常")
torch.testing.assert_close(safe_softmax(x_large), torch.softmax(x_large, -1))
print("safe softmax 与 torch.softmax 一致 ✓ (softmax 平移不变性)")

# fp16 更脆弱:x > 11.1 就爆
x16 = (torch.randn(4, 8, device="cuda") + 12).half()
print("fp16, x≈12: naive 含 NaN?", naive_softmax(x16).isnan().any().item())
```

### 例 2:online softmax 的纯 Python 参考实现 —— 逐块验证恒等式

先在 Python 里把算法写对、看懂,再搬进 kernel——这是开发数值算法 kernel 的标准流程(也是第 14 篇的开发路径)。

```python
def online_softmax_reference(row: torch.Tensor, block: int):
    """流式扫描一行,只维护 (m, d) 两个标量。返回与 softmax 相同的结果。"""
    m = float("-inf")   # 运行最大值
    d = 0.0             # 以 m 为基准的指数和
    # ---- 第一遍:流式统计 ----
    for k in range(0, len(row), block):
        blk = row[k:k + block]
        m_new = max(m, blk.max().item())
        d = d * torch.exp(torch.tensor(m - m_new)).item() + torch.exp(blk - m_new).sum().item()
        m = m_new
    # ---- 第二遍:归一化输出 ----
    return torch.exp(row - m) / d


row = torch.randn(10_000, device="cuda") * 10   # 长行 + 大动态范围
for block in [128, 1000, 4096]:                  # 不同块大小,结果应完全一致
    out = online_softmax_reference(row, block)
    torch.testing.assert_close(out, torch.softmax(row, -1), rtol=1e-5, atol=1e-7)
print("online softmax 在任意分块下都与全量 softmax 精确一致 ✓ (它是精确算法,不是近似)")
```

### 例 3:Triton 实现 —— 能吃任意行宽的 softmax kernel

每行一个 program,行内用 `for` 循环按 BLOCK 分块:第一趟 online 统计,第二趟归一化写出。与第 07 篇单块版的本质区别:**BLOCK 是固定小常量,行宽 N 是运行期变量,多长都不怕**。

```python
@triton.jit
def online_softmax_kernel(x_ptr, out_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    row_start = x_ptr + row * stride_m

    # ---- 第一趟:流式求 (m, d)。m/d 是标量,活在寄存器里 ----
    m = float("-inf")
    d = 0.0
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        cols = k * BLOCK_N + tl.arange(0, BLOCK_N)
        blk = tl.load(row_start + cols, mask=cols < N, other=float("-inf"))
        blk_max = tl.max(blk, axis=0)
        m_new = tl.maximum(m, blk_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(blk - m_new), axis=0)
        m = m_new

    # ---- 第二趟:归一化写出 ----
    out_start = out_ptr + row * stride_m
    for k in range(0, tl.cdiv(N, BLOCK_N)):
        cols = k * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        blk = tl.load(row_start + cols, mask=mask, other=0.0)
        tl.store(out_start + cols, tl.exp(blk - m) / d, mask=mask)


def online_softmax(x, BLOCK_N=2048):
    M, N = x.shape
    out = torch.empty_like(x)
    online_softmax_kernel[(M,)](x, out, M, N, x.stride(0), BLOCK_N=BLOCK_N, num_warps=8)
    return out


# 行宽 100_000:第 07 篇的单块方案在这里直接出局
x = torch.randn(64, 100_000, device="cuda") * 5
torch.testing.assert_close(online_softmax(x), torch.softmax(x, -1), rtol=1e-5, atol=1e-7)
print("行宽 100,000 的 softmax 正确 ✓ —— 单块容量不再是限制")
```

---

## §3 知识连接

**与前面篇章:**

- 第 07 篇遗留问题(单块装不下长行)在例 3 正式解决;那篇的 `other=-inf / 0` 填充规则在这里继续守护边界;
- 第 06 篇"计算换存储"的母题再现:online softmax 多做几次乘法(换基准),换走一整遍显存读写;
- 第 08 篇方法论落地:例 2 先用 Python 参考实现验证算法,再写 kernel 对照——复杂 kernel 的标准开发流程。

**与 FlashAttention 的关系(第 14 篇预告,面试核心):**

attention 的 $O = \text{softmax}(QK^T)V$ 中,softmax 的输入是 $N \times N$ 的 score 矩阵——长序列下根本不想把它物化到显存。Online softmax 提供的能力恰好是:**score 一块一块地算出来,统计量流式更新,算完即扔**。FlashAttention 在 $(m, d)$ 之外再维护一个"以当前基准缩放过的输出累加器" $\tilde{O}$,换基准时同步乘 $e^{m_{old}-m_{new}}$ ——仅此而已。把本篇例 2 看熟,第 14 篇的推导就是显然的。

**与真实框架:**

- 原始论文:Milakov & Gimelshein, *Online normalizer calculation for softmax*(2018);FlashAttention(Dao et al. 2022)将其与 tiling 结合;
- PyTorch 的 `torch.softmax` 对长行内部也用分块归约(但仍物化中间结果);`F.scaled_dot_product_attention` 的 memory-efficient 后端用的就是 online 思想;
- vLLM/SGLang 的 attention kernel、Liger-Kernel 的 fused cross-entropy(第 23 篇)里都能找到 `m_new = max(m, ...); acc *= exp(m - m_new)` 这个签名式的代码片段——认识它,源码就读懂一半。

---

## §4 闭环对比实验:三种 softmax 策略 × 行宽扫描

选手:① 第 07 篇单块 kernel(行宽受限);② 本篇 online kernel;③ `torch.softmax`。行宽从 4K 扫到 128K(总元素量固定 ~3200 万)。单块版在装不下时会编译失败/资源不足——我们如实捕获并标记。

```python
import matplotlib.pyplot as plt

@triton.jit
def singleblock_softmax_kernel(x_ptr, out_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    v = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=float("-inf"))
    v = v - tl.max(v, axis=0)
    num = tl.exp(v)
    tl.store(out_ptr + row * stride_m + cols, num / tl.sum(num, axis=0), mask=mask)


def singleblock_softmax(x):
    M, N = x.shape
    out = torch.empty_like(x)
    singleblock_softmax_kernel[(M,)](x, out, M, N, x.stride(0),
                                     BLOCK_N=triton.next_power_of_2(N), num_warps=16)
    return out


widths = [4096, 16384, 32768, 65536, 131072]
total = 32 * 1024 * 1024
res = {"single-block (07)": [], "online (09)": [], "torch.softmax": []}

for N in widths:
    M = max(total // N, 8)
    t = torch.randn(M, N, device="cuda")
    bytes_moved = 2 * M * N * 4

    # 单块版:可能因资源不足失败
    try:
        torch.testing.assert_close(singleblock_softmax(t), torch.softmax(t, -1))
        ms = triton.testing.do_bench(lambda: singleblock_softmax(t), return_mode="median")
        res["single-block (07)"].append(bytes_moved / (ms / 1000) / 1e9)
    except Exception as exc:
        res["single-block (07)"].append(float("nan"))
        print(f"N={N}: 单块版失败({type(exc).__name__})—— 这就是它的能力边界")

    torch.testing.assert_close(online_softmax(t), torch.softmax(t, -1), rtol=1e-5, atol=1e-7)
    ms = triton.testing.do_bench(lambda: online_softmax(t), return_mode="median")
    res["online (09)"].append(bytes_moved / (ms / 1000) / 1e9)

    ms = triton.testing.do_bench(lambda: torch.softmax(t, -1), return_mode="median")
    res["torch.softmax"].append(bytes_moved / (ms / 1000) / 1e9)

print(f"\n{'行宽':>8} | " + " | ".join(f"{k:>18}" for k in res))
for i, N in enumerate(widths):
    print(f"{N:>8} | " + " | ".join(f"{res[k][i]:>13.1f} GB/s" for k in res))

plt.figure(figsize=(9, 4.5))
for name, vals in res.items():
    plt.semilogx(widths, vals, "o-", label=name)
plt.axhline(320, color="gray", ls="--", label="T4 320 GB/s")
plt.xlabel("row width N")
plt.ylabel("effective bandwidth (GB/s)")
plt.title("softmax strategies vs row width")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 实验结果解读

- **单块版**:中等行宽性能尚可,行宽增大后先性能跳水(寄存器溢出到 local memory)、最终直接无法运行——清晰的能力边界演示;
- **online 版**:全行宽稳定工作,带宽贴近上限;两遍读一遍写(流量 3n×4B,理论上限因此是单遍版的 2/3 左右)——精确性与普适性的代价;
- `torch.softmax` 内部也做了分块,长行下与 online 版接近;
- **真正的赢家在下一阶段**:attention 场景里 score 矩阵根本不需要写出来,online 统计直接融进 matmul 流水线,两遍变一遍、写出量从 $O(N^2)$ 变 $O(N)$——第 14 篇 FlashAttention 收割本篇种下的一切。

---

## §5 练习 + 面试考点

### 动手练习

1. 实现 fused `log_softmax`(交叉熵的前半段):利用 $\log\text{softmax}(x) = x - m - \log d$,只需第一趟统计 + 第二趟写出,与 `torch.log_softmax` 对照。这是第 23 篇 fused cross-entropy 的热身。
2. 把例 3 的第一趟改成**并行分段**:每行启动 4 个 program 各算一段的 $(m_k, d_k)$,再用合并公式 $m = \max(m_a, m_b),\ d = d_a e^{m_a - m} + d_b e^{m_b - m}$ 归并(可在 torch 侧完成)。验证正确性——你刚刚实现了 Flash-Decoding 的核心思想(第 17 篇)。

### 面试高频考点

- **Q:softmax 为什么要减 max?减任意常数行不行?**
  A:防止 exp 上溢(fp32 在 88.7、fp16 在 11.1 就爆);softmax 有平移不变性,减任何常数数学等价,但减 max 能保证指数 ≤ 0 永不上溢,同时分母 ≥ 1 也不会下溢成 0。
- **Q:online softmax 的更新公式?为什么是精确的?**
  A:$m \leftarrow \max(m, m_{blk})$,$d \leftarrow d\,e^{m_{old}-m_{new}} + \sum e^{x_{blk}-m_{new}}$。换基准因子 $e^{m_{old}-m_{new}}$ 把旧分母无损转换到新基准,代数恒等,无近似。
- **Q:FlashAttention 和 online softmax 什么关系?**
  A:FlashAttention = tiling(分块算 $QK^T$)+ online softmax(流式归一化统计)+ 输出累加器同步换基准。没有 online softmax,分块算 attention 就必须存下整个 score 矩阵,IO 优势不复存在。
- **Q:online softmax 的统计量合并满足什么性质?有什么用?**
  A:结合律与交换律——$(m,d)$ 对的合并构成幺半群。因此分块既可以串行流式,也可以并行分段再归并:后者就是 Flash-Decoding 对超长 KV 做 split-K 并行的数学基础。
