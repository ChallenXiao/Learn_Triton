# 12 · Norm 类算子:LayerNorm、RMSNorm 与 GEMM epilogue 融合

> Learn Triton 系列 · 阶段 2(核心算子)第 4 篇
> 前置:第 07 篇(行归约模式)、第 10/11 篇(GEMM)
> 运行环境:Google Colab T4 GPU

Transformer 每一层都要做两次规范化(attention 前、MLP 前),序列越长调用越密——Norm 类算子是推理框架里被融合得最狠的一类。本篇做两件事:① 用第 07 篇的行归约模式写出 LayerNorm / RMSNorm(现代 LLM 的标配);② 引入**第三级融合形态:GEMM epilogue 融合**——把 bias、激活直接挂在 matmul 的累加器后面,中间结果一个字节都不落显存。

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

### LayerNorm 与 RMSNorm

对隐藏向量 $x \in \mathbb{R}^N$(一行):

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta, \qquad \mu = \tfrac{1}{N}\sum x_i,\ \sigma^2 = \tfrac{1}{N}\sum (x_i - \mu)^2$$

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{N}\sum x_i^2 + \epsilon}} \cdot \gamma$$

RMSNorm(Llama/Qwen/Mistral 全在用)去掉了减均值和 $\beta$:少一次归约、少一个参数,效果几乎不变——**典型的"为硬件效率简化算法"**。

计算模式 = 第 07 篇模式 A:每行一个 program,一次 load,行内归约出统计量,逐元素变换写出。理想流量:读 $4N$ + 写 $4N$ 字节(fp32),eager 实现(分解为 mean/var/sub/div/mul/add 多个 kernel)流量是它的 3~5 倍。

### Epilogue 融合:挂在 GEMM 尾巴上的免费午餐

第 10 篇 GEMM 的累加器 `acc` 在写出前就躺在寄存器里。`Linear + bias + activation` 这种模式,eager 要 3 个 kernel、读写 $MN$ 矩阵 3 次;而在 Triton 里只需在 `tl.store` 之前插两行——**bias 和激活的成本约等于零**(GEMM 是 compute-bound,多几条逐元素指令不动瓶颈)。cuBLASLt 内置了少量固定 epilogue(bias+relu/gelu),Triton 的优势是 epilogue **任意可编程**(量化、残差、甚至随机数)。

### 能做什么

- LayerNorm/RMSNorm 前向:单 kernel、贴近带宽上限,对 Transformer 常见行宽(768~16384)全覆盖;
- 统计量(mean/rstd)可顺手写出,供反向使用(第 13 篇);
- epilogue 里可以做:bias、任意激活、残差相加、类型转换、per-channel 缩放(第 21 篇量化反缩放就挂在这);
- fp16 输入 + fp32 内部统计:精度与速度兼得(kernel 内 `.to(tl.float32)`)。

### 不能做什么

- 行宽超过单块容量时本篇写法失效(需换第 09 篇 online/两遍式;Welford 流式方差见练习);
- epilogue 融合**只能向后融合到本次写出为止**:跨 GEMM 的融合(Linear→Norm→Linear)做不到,因为中间有跨行归约,必须断 kernel(这正是 Norm 单独成 kernel 的原因);
- LayerNorm 的**反向**需要保存统计量并做两次跨行归约,比前向难一个档次(第 13 篇正面处理);
- $E[x^2] - E[x]^2$ 公式在大均值数据上有灾难性消除风险——本篇用"先减均值"的两步法保数值安全,流式场景用 Welford。

---

## §2 递进式例子

### 例 1:LayerNorm 前向 kernel

```python
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, stride_m, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N                       # 归约 1:均值
    diff = tl.where(mask, x - mean, 0.0)               # 越界位必须清零,否则污染方差
    var = tl.sum(diff * diff, axis=0) / N              # 归约 2:方差(先减均值,数值安全)
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * w + b
    tl.store(out_ptr + row * stride_m + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


def triton_layernorm(x, w, b, eps=1e-5):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    layernorm_kernel[(M,)](x, w, b, out, M, N, x.stride(0), eps,
                           BLOCK_N=BLOCK_N, num_warps=8 if BLOCK_N >= 2048 else 4)
    return out


M, N = 4096, 4096
x = torch.randn(M, N, device="cuda", dtype=torch.float16)
w = torch.randn(N, device="cuda", dtype=torch.float16)
b = torch.randn(N, device="cuda", dtype=torch.float16)
torch.testing.assert_close(triton_layernorm(x, w, b), F.layer_norm(x, (N,), w, b), rtol=1e-2, atol=1e-2)
print("LayerNorm 正确 ✓ (fp16 输入, fp32 内部统计)")
print("注意 diff 那行的 tl.where:mask 外填充的 0 经过 x-mean 会变成 -mean,不清零就污染方差 —— 高频 bug")
```

### 例 2:RMSNorm kernel —— Llama 系标配

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, M, N, stride_m, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)     # 只有一次归约,且不用担心填充值(0²=0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_m + cols, (x / rms * w).to(x_ptr.dtype.element_ty), mask=mask)


def triton_rmsnorm(x, w, eps=1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    rmsnorm_kernel[(M,)](x, w, out, M, N, x.stride(0), eps,
                         BLOCK_N=BLOCK_N, num_warps=8 if BLOCK_N >= 2048 else 4)
    return out


def rmsnorm_ref(x, w, eps=1e-6):
    """与 Llama 官方实现一致的参考版本(fp32 内部计算)。"""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


torch.testing.assert_close(triton_rmsnorm(x, w), rmsnorm_ref(x, w), rtol=1e-2, atol=1e-2)
print("RMSNorm 正确 ✓ —— 比 LayerNorm 少一次归约、少一个参数,这就是 Llama 选它的理由")
```

### 例 3:GEMM epilogue 融合 —— Linear + bias + GELU 一个 kernel

在第 10 篇 kernel 的 `tl.store` 前插入 epilogue。对照组是 eager 的三连击。

```python
@triton.jit
def matmul_bias_gelu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ======= epilogue:成本趋近于零的三行 =======
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]                                   # + bias(沿行广播)
    inner = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
    acc = 0.5 * acc * (1.0 + (2.0 * tl.sigmoid(2.0 * inner) - 1.0))   # GELU(tanh)
    # ===========================================

    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def fused_linear_gelu(a, b, bias):
    M, K = a.shape; N = b.shape[1]
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    matmul_bias_gelu_kernel[grid](a, b, bias, c, M, N, K,
                                  a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                  c.stride(0), c.stride(1),
                                  BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                                  num_warps=4, num_stages=4)
    return c


M, K, N = 2048, 2048, 2048
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
bw_ = torch.randn(K, N, device="cuda", dtype=torch.float16)
bias = torch.randn(N, device="cuda", dtype=torch.float16)
ref = F.gelu((a @ bw_) + bias, approximate="tanh")
torch.testing.assert_close(fused_linear_gelu(a, bw_, bias), ref, rtol=2e-2, atol=2e-2)
print("matmul+bias+GELU 单 kernel 正确 ✓")
```

---

## §3 知识连接

**与前面篇章:**

- 例 1/2 = 第 07 篇模式 A + 仿射变换;例 1 的 `tl.where(mask, x-mean, 0)` 是第 04 篇填充值问题的第三次现身(sum 填 0 / max 填 -inf / **减均值后要重新清零**);
- 例 3 = 第 10 篇 GEMM + 第 06 篇逐元素融合,两条线在此汇合。融合的四级阶梯现在凑齐三级:逐元素(06)→ 归约+逐元素(07/12)→ GEMM epilogue(12)→ 下一站 GEMM+归约+GEMM(14,FlashAttention);
- fp16 进、fp32 算、fp16 出的精度纪律与第 10 篇累加器一脉相承。

**与真实框架:**

- vLLM:`fused_add_rms_norm`(残差相加 + RMSNorm 一个 kernel,第 16 篇我们实现它)是每个 decoder layer 调两次的热点;
- Liger-Kernel:`liger_kernel/ops/rms_norm.py` 与例 2 几乎逐行对应(它多了反向);其 `fused_linear_*` 系列就是例 3 思想 + 反向;
- PyTorch:`F.layer_norm` 走 cuDNN/ATen 的专用 kernel(已经融合);`nn.Linear + GELU` 在 eager 下不融合,torch.compile 能融 bias+gelu 进 matmul 的 epilogue(Inductor 调 cuBLASLt epilogue 或生成 Triton GEMM)——实验里见分晓。

---

## §4 闭环对比实验

### 实验 A:RMSNorm 三方对决(memory-bound,看带宽)

```python
import matplotlib.pyplot as plt

compiled_rmsnorm = torch.compile(rmsnorm_ref)

M = 8192
widths = [1024, 2048, 4096, 8192]
res = {"Triton": [], "eager (分解)": [], "torch.compile": []}
for N in widths:
    t = torch.randn(M, N, device="cuda", dtype=torch.float16)
    g = torch.randn(N, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(triton_rmsnorm(t, g), rmsnorm_ref(t, g), rtol=1e-2, atol=1e-2)
    bytes_moved = 2 * M * N * 2
    for name, fn in [("Triton", lambda: triton_rmsnorm(t, g)),
                     ("eager (分解)", lambda: rmsnorm_ref(t, g)),
                     ("torch.compile", lambda: compiled_rmsnorm(t, g))]:
        ms = triton.testing.do_bench(fn, return_mode="median")
        res[name].append(bytes_moved / (ms / 1000) / 1e9)

print(f"{'隐层宽度':>8} | " + " | ".join(f"{k:>14}" for k in res))
for i, N in enumerate(widths):
    print(f"{N:>8} | " + " | ".join(f"{res[k][i]:>10.1f} GB/s" for k in res))
```

### 实验 B:epilogue 融合 vs eager 三连 vs torch.compile

```python
def eager_chain(a, b, bias):
    return F.gelu((a @ b) + bias, approximate="tanh")

compiled_chain = torch.compile(eager_chain)

flops = 2 * M * K * N
impls = {
    "eager: matmul;+bias;gelu": lambda: eager_chain(a, bw_, bias),
    "torch.compile": lambda: compiled_chain(a, bw_, bias),
    "Triton epilogue 融合": lambda: fused_linear_gelu(a, bw_, bias),
}
print(f"\n{'实现':28s} {'中位数':>10} {'TFLOPS':>8}")
base = None
for name, fn in impls.items():
    torch.testing.assert_close(fn(), ref, rtol=2e-2, atol=2e-2)
    ms = triton.testing.do_bench(fn, return_mode="median")
    base = base or ms
    print(f"{name:28s} {ms:9.4f}ms {flops / (ms/1000) / 1e12:>7.2f} ({base/ms:.2f}x)")

plt.figure(figsize=(9, 4))
for name, vals in res.items():
    plt.plot(widths, vals, "o-", label=name)
plt.axhline(320, color="gray", ls="--", label="T4 320 GB/s")
plt.xlabel("hidden width N"); plt.ylabel("GB/s")
plt.title("RMSNorm bandwidth (experiment A)")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

### 实验结果解读

- **实验 A**:eager 分解版(pow/mean/rsqrt/mul/mul 五连 kernel)带宽折损 3 倍以上;Triton 单 kernel 贴近带宽上限;compile 居中或追平——归约+逐元素的融合编译器已基本掌握,但手写仍稳定占优;
- **实验 B**:epilogue 融合相对 eager 三连有 10%~25% 的整体提升——看似不大,但这是**白拿的**:GELU 和 bias 的字节流量被完全消除,GEMM 本体一纳秒没变慢。形状越偏 memory-bound(小 M/N),epilogue 融合收益越大;
- 工程启示:模型里每个 `Linear → 激活` 都值得检查是否融合(torch.compile 大多能做);而 `Linear → RMSNorm` 之间必须断 kernel(跨行归约),所以 Norm 自身的高效实现是独立战场。

---

## §5 练习 + 面试考点

### 动手练习

1. 实现 **fused residual + RMSNorm**:`y = rmsnorm(x + residual)`,同时把 `x + residual` 也写出(下一层还要用)。这是 vLLM `fused_add_rms_norm` 的精确复刻,第 16 篇会用到——先自己试。
2. 把例 1 改成 **Welford 流式方差** 的分块版本(行宽大于单块时也能算),与 `E[x²]-E[x]²` 公式版在 `x ~ N(1000, 1)` 的数据上比较精度,体会灾难性消除。

### 面试高频考点

- **Q:RMSNorm 和 LayerNorm 的区别?为什么新模型都用 RMSNorm?**
  A:RMSNorm 去掉减均值与 β,只按均方根缩放。少一次归约、参数减半、训练稳定性相当(论文与 Llama 实践验证)。kernel 角度:LayerNorm 两次归约 + 减均值后需重新 mask,RMSNorm 一次归约且 0 填充天然安全,更好写也更快。
- **Q:为什么 Norm 前后的算子不能与 Norm 融成更大的 kernel?**
  A:Norm 含跨整行的归约,行的全部元素必须先就位;前面的 GEMM 按瓦片产出、后面的 GEMM 按瓦片消费,瓦片边界与行归约冲突,kernel 必须在归约处断开。能融的只有"行内"邻居:残差相加、激活、量化缩放。
- **Q:什么是 epilogue 融合?它为什么几乎免费?**
  A:GEMM 累加器写出前,在寄存器上顺手完成 bias/激活/缩放等逐元素操作。GEMM 是 compute-bound,瓶颈在 Tensor Core 流水线;epilogue 增加的少量标量指令不在关键路径上,却省掉了独立 kernel 的整轮 MN 矩阵读写。
- **Q:fp16 模型里 Norm 为什么内部要转 fp32?**
  A:方差是平方和,fp16 最大 65504,4096 个 x²≈1 的数求和就接近上限,极易溢出;均值的舍入误差也会被 rsqrt 放大。读写用 fp16(省带宽)、统计量用 fp32(保精度)是业界统一做法(vLLM/Liger/cuDNN 皆如此)。
