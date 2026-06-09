# 16 · 推理高频小算子:RoPE、SwiGLU 与 fused residual+RMSNorm

> Learn Triton 系列 · 阶段 3(推理优化)第 2 篇
> 前置:第 06 篇(逐元素融合)、第 12 篇(RMSNorm)、第 15 篇(decode 是 memory-bound)
> 运行环境:Google Colab T4 GPU

第 15 篇的图谱说:decode 阶段每个字节都是延迟。一个 Llama 风格的 decoder layer 里,除了 Linear 和 attention,还散落着一堆"小算子":旋转位置编码(RoPE)、门控激活(SwiGLU)、残差与规范化。它们单看都不大,但 eager 模式下每个都是独立 kernel、独立显存往返,**在 memory-bound 的 decode 里聚沙成塔**。本篇把 vLLM 里出镜率最高的三个融合 kernel 亲手写一遍,最后做一个"整个 decoder block 算子级替换"的闭环实验。

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

### 一个 Llama decoder layer 的算子清单

```text
x ──► RMSNorm ──► QKV Linear ──► RoPE(q,k) ──► Attention ──► O Linear ──► (+residual)
  └────────────────────────────────────────────────────────────────────────┘
   ──► RMSNorm ──► Gate/Up Linear ──► SwiGLU ──► Down Linear ──► (+residual)
```

Linear 走 cuBLAS、attention 走 FlashAttention 之后,剩下的就是本篇三件套。它们全是 memory-bound 逐元素/行归约类,优化手段只有一个:**融合,减字节**。

### RoPE(旋转位置编码)

把位置信息编码成对 q/k 向量的**旋转**:head_dim 维向量按"前半 vs 后半"配对成 $d/2$ 个二维平面,第 $i$ 对在位置 $p$ 旋转角度 $p\cdot\theta_i$,$\theta_i = \text{base}^{-2i/d}$:

$$\begin{pmatrix} x'_i \\ x'_{i+d/2} \end{pmatrix} = \begin{pmatrix} \cos p\theta_i & -\sin p\theta_i \\ \sin p\theta_i & \cos p\theta_i \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+d/2} \end{pmatrix}$$

性质:$\langle R_p q, R_s k \rangle$ 只依赖相对位置 $p-s$ ——这是它取代绝对位置编码的原因。**kernel 视角**:纯逐元素操作(每对独立),但要按 position 查 cos/sin 表(gather,第 05 篇),且 q、k 两个张量可在一个 kernel 内同时处理。

### SwiGLU

Llama MLP 的激活:`down( silu(gate(x)) * up(x) )`。gate 和 up 两个 Linear 之后,`silu(g) * u` 是个三算子链(sigmoid、mul、mul),eager 下 2~3 个 kernel、读写 5 个大张量;融合后读 2 写 1。

### fused residual + RMSNorm

decoder layer 里出现两次的模式:`residual = x + residual; y = rmsnorm(residual)`。融合 kernel 一次 load 完成加法 + 归约 + 缩放,**同时写出两个张量**(y 给下一个 Linear,新 residual 给下一次相加)——多输出融合(第 06 篇能力清单)的实战。

### 能做什么 / 不能做什么

能做:三件套各自省 2~4 倍小算子流量;decode 形状下(batch×1 token)还能省可观的 kernel 启动开销(第 01 篇);RoPE 可以顺手做 **Neox/GPT-J 两种配对约定**、缩放(NTK/YaRN)等变体——改几行的事,这是手写的灵活性。

不能做:

- 改变大头:Linear 与 attention 占 decoder layer 时间的 70%~90%,三件套优化的是剩下那块(实验里量化这一点,管理预期);
- RoPE 的 cos/sin 表通常预计算缓存;在 kernel 里现场 `tl.cos/tl.sin` 也行,但精度/速度权衡要实测(本篇用查表,与 vLLM 一致);
- 不同模型的 RoPE 配对约定不同(Llama half-split vs GPT-J interleaved):**kernel 写死一种,用错即静默出错**——必须用参考实现对拍。

---

## §2 递进式例子

### 例 1:RoPE kernel —— 同时旋转 q 和 k

布局 `[T, H, D]`(T 个 token、H 个头)。grid = (T, H),每个 program 旋转一个头的向量;按 `pos_ptr` 查表(decode 时各请求位置不同,必须显式传)。

```python
@triton.jit
def rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, pos_ptr,
                T, H_q, H_k, D: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)         # 第几个 token
    h = tl.program_id(1)         # 第几个头(q 和 k 头数可不同 -> GQA)
    pos = tl.load(pos_ptr + t)   # 这个 token 的位置(gather)

    half = D // 2
    i = tl.arange(0, BLOCK_D)    # 0..half-1
    mask = i < half
    cos = tl.load(cos_ptr + pos * half + i, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + pos * half + i, mask=mask, other=0.0)

    # ---- 旋转 q(仅当 h < H_q)----
    if h < H_q:
        base = q_ptr + (t * H_q + h) * D
        x1 = tl.load(base + i, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(base + half + i, mask=mask, other=0.0).to(tl.float32)
        tl.store(base + i, (x1 * cos - x2 * sin).to(tl.float16), mask=mask)
        tl.store(base + half + i, (x2 * cos + x1 * sin).to(tl.float16), mask=mask)
    # ---- 旋转 k(仅当 h < H_k,GQA 下 k 头更少)----
    if h < H_k:
        base = k_ptr + (t * H_k + h) * D
        x1 = tl.load(base + i, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(base + half + i, mask=mask, other=0.0).to(tl.float32)
        tl.store(base + i, (x1 * cos - x2 * sin).to(tl.float16), mask=mask)
        tl.store(base + half + i, (x2 * cos + x1 * sin).to(tl.float16), mask=mask)


def build_rope_cache(max_pos, head_dim, base=10000.0):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2, device="cuda").float() / head_dim)
    t = torch.arange(max_pos, device="cuda").float()
    freqs = torch.outer(t, inv)                       # [max_pos, head_dim/2]
    return freqs.cos(), freqs.sin()


def rope_ref(x, cos, sin, pos):
    """Llama half-split 参考实现。x: [T, H, D]"""
    c = cos[pos][:, None, :].float()
    s = sin[pos][:, None, :].float()
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2].float(), x[..., d2:].float()
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).to(x.dtype)


T, Hq, Hk, D = 4096, 14, 2, 64                       # Qwen2.5-0.5B 的头配置(GQA)
cos, sin = build_rope_cache(8192, D)
pos = torch.randint(0, 8192, (T,), device="cuda", dtype=torch.int64)
q = torch.randn(T, Hq, D, device="cuda", dtype=torch.float16)
k = torch.randn(T, Hk, D, device="cuda", dtype=torch.float16)
q_ref, k_ref = rope_ref(q, cos, sin, pos), rope_ref(k, cos, sin, pos)

rope_kernel[(T, max(Hq, Hk))](q, k, cos, sin, pos, T, Hq, Hk,
                              D=D, BLOCK_D=triton.next_power_of_2(D // 2))
torch.testing.assert_close(q, q_ref, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(k, k_ref, rtol=1e-3, atol=1e-3)
print("RoPE 正确 ✓(in-place 旋转 q/k,GQA 头数不对称,按 position 查表)")
```

### 例 2:SwiGLU 融合 kernel

```python
@triton.jit
def swiglu_kernel(g_ptr, u_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, (g * tl.sigmoid(g) * u).to(tl.float16), mask=mask)


def triton_swiglu(g, u):
    out = torch.empty_like(g)
    n = g.numel()
    swiglu_kernel[(triton.cdiv(n, 1024),)](g, u, out, n, BLOCK=1024)
    return out


g = torch.randn(4096, 4864, device="cuda", dtype=torch.float16)
u = torch.randn_like(g)
torch.testing.assert_close(triton_swiglu(g, u), F.silu(g.float()).half() * u, rtol=1e-2, atol=1e-2)
print("SwiGLU 正确 ✓(eager: sigmoid+mul+mul 三 kernel 五次读写 -> 融合: 一 kernel 读2写1)")
```

### 例 3:fused residual + RMSNorm —— 双输出融合(第 12 篇练习的答案)

```python
@triton.jit
def fused_add_rmsnorm_kernel(x_ptr, res_ptr, w_ptr, y_ptr, res_out_ptr,
                             M, N, stride_m, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)

    s = x + r                                                   # 残差相加
    tl.store(res_out_ptr + row * stride_m + cols, s.to(tl.float16), mask=mask)  # 输出 1:新残差

    rstd = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / N + eps)       # 行归约
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_m + cols, (s * rstd * w).to(tl.float16), mask=mask)  # 输出 2:norm 结果


def fused_add_rmsnorm(x, res, w, eps=1e-6):
    M, N = x.shape
    y, res_out = torch.empty_like(x), torch.empty_like(x)
    fused_add_rmsnorm_kernel[(M,)](x, res, w, y, res_out, M, N, x.stride(0), eps,
                                   BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return y, res_out


def add_rmsnorm_ref(x, res, w, eps=1e-6):
    s = (x.float() + res.float())
    y = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).half(), s.half()


x = torch.randn(4096, 896, device="cuda", dtype=torch.float16)
res = torch.randn_like(x)
w = torch.randn(896, device="cuda", dtype=torch.float16)
y1, r1 = fused_add_rmsnorm(x, res, w)
y2, r2 = add_rmsnorm_ref(x, res, w)
torch.testing.assert_close(y1, y2, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(r1, r2, rtol=1e-2, atol=1e-2)
print("fused add+RMSNorm 正确 ✓(一次 load,两个输出;与 vLLM fused_add_rms_norm 同构)")
```

---

## §3 知识连接

**与前面篇章:**

- 三件套分别是前面模式的实战化:RoPE = 逐元素 + gather 查表(05/06);SwiGLU = 纯逐元素融合(06);fused add+RMSNorm = 行归约 + 多输出(07/12);
- 例 1 的 GQA 头数不对称(Hq=14, Hk=2)呼应第 15 篇例 2 的发现,第 17 篇正式展开;
- decode 形状下这些 kernel 的输入很小(batch×1 token),第 01 篇的启动开销占比回升——融合同时省字节和启动次数,双重收益。

**与真实框架(本篇 kernel 的"原型出处"):**

- vLLM:`vllm/model_executor/layers/rotary_embedding/`(RoPE,支持十几种变体)、`_custom_ops.silu_and_mul`(SwiGLU,gate/up 拼在一个张量里切半处理)、`fused_add_rms_norm`(例 3);早期均为 CUDA 实现,Triton 版本在 `vllm/model_executor/layers/` 各处可见;
- SGLang `sgl-kernel` 与 Liger-Kernel(`liger_kernel/ops/rope.py`、`swiglu.py`、`rms_norm.py`)有这三件套的 Triton 实现,接口和本篇几乎一致——**学完本篇可以直接去读/改这些生产代码**;
- HuggingFace transformers 的 eager 实现(`modeling_llama.py` 的 `apply_rotary_pos_emb`、`LlamaMLP`)就是本篇各 ref 函数的出处。

---

## §4 闭环对比实验:对一个 Llama 风格 block 做算子级替换

搭一个真实尺寸的 decoder block(Qwen2.5-0.5B 配置:hidden 896、FFN 4864、14/2 头 GQA)。对照组全 eager;实验组把三件套换成 Triton kernel(Linear 仍走 cuBLAS、attention 仍走 SDPA——只动我们该动的)。分别测 **prefill 形状**(4×1024 token)与 **decode 形状**(batch 64×1 token)。

```python
import matplotlib.pyplot as plt

H, FFN, HQ, HKV, D = 896, 4864, 14, 2, 64

class Block(torch.nn.Module):
    def __init__(self, use_triton):
        super().__init__()
        self.use_triton = use_triton
        self.norm1_w = torch.nn.Parameter(torch.ones(H))
        self.norm2_w = torch.nn.Parameter(torch.ones(H))
        self.qkv = torch.nn.Linear(H, (HQ + 2 * HKV) * D, bias=False)
        self.o = torch.nn.Linear(HQ * D, H, bias=False)
        self.gate = torch.nn.Linear(H, FFN, bias=False)
        self.up = torch.nn.Linear(H, FFN, bias=False)
        self.down = torch.nn.Linear(FFN, H, bias=False)

    def forward(self, x, res, cos, sin, pos):
        T = x.shape[0]
        # ---- attention 半区 ----
        if self.use_triton:
            h, res = fused_add_rmsnorm(x, res, self.norm1_w)
        else:
            h, res = add_rmsnorm_ref(x, res, self.norm1_w)
        qkv = self.qkv(h)
        q, k, v = qkv.split([HQ * D, HKV * D, HKV * D], dim=-1)
        q, k, v = q.view(T, HQ, D), k.view(T, HKV, D), v.view(T, HKV, D)
        if self.use_triton:
            rope_kernel[(T, HQ)](q, k, cos, sin, pos, T, HQ, HKV, D=D,
                                 BLOCK_D=triton.next_power_of_2(D // 2))
        else:
            q, k = rope_ref(q, cos, sin, pos), rope_ref(k, cos, sin, pos)
        k = k.repeat_interleave(HQ // HKV, dim=1)     # GQA -> MHA 展开(教学简化)
        v = v.repeat_interleave(HQ // HKV, dim=1)
        a = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        x = self.o(a.transpose(0, 1).reshape(T, HQ * D))
        # ---- MLP 半区 ----
        if self.use_triton:
            h, res = fused_add_rmsnorm(x, res, self.norm2_w)
            m = triton_swiglu(self.gate(h), self.up(h))
        else:
            h, res = add_rmsnorm_ref(x, res, self.norm2_w)
            m = F.silu(self.gate(h).float()).half() * self.up(h)
        return self.down(m), res


cos, sin = build_rope_cache(8192, D)
results = {}
for shape_name, T in [("prefill 4x1024", 4096), ("decode batch=64", 64)]:
    x = torch.randn(T, H, device="cuda", dtype=torch.float16)
    res0 = torch.zeros_like(x)
    pos = torch.arange(T, device="cuda") % 1024
    outs = {}
    for name, ut in [("eager", False), ("Triton 三件套", True)]:
        blk = Block(ut).cuda().half()
        if outs:                                       # 用同一套权重保证可比
            blk.load_state_dict(prev_state)
        prev_state = blk.state_dict()
        fn = lambda: blk(x, res0.clone(), cos, sin, pos)
        out, _ = fn()
        outs[name] = out
        ms = triton.testing.do_bench(fn, return_mode="median")
        results[f"{shape_name} | {name}"] = ms
        print(f"{shape_name:>18s} | {name:12s} {ms:8.3f} ms")
    torch.testing.assert_close(outs["eager"], outs["Triton 三件套"], rtol=3e-2, atol=3e-2)
    sp = results[f"{shape_name} | eager"] / results[f"{shape_name} | Triton 三件套"]
    print(f"{'':18s} -> block 整体加速 {sp:.2f}x\n")

plt.figure(figsize=(9, 4))
names = list(results.keys())
plt.barh(names[::-1], [results[n] for n in names[::-1]])
plt.xlabel("block latency (ms)"); plt.title("Llama-style block: eager vs Triton ops")
plt.tight_layout(); plt.show()
```

### 实验结果解读

- **decode 形状收益大于 prefill 形状**:decode 下小算子的启动开销与碎片化读写占比更高,融合的双重收益(字节+启动)都兑现;prefill 下 GEMM 占大头,三件套的提速被稀释——第 15 篇图谱的预言落地;
- block 整体加速通常在 1.1~1.4 倍量级,**而不是单算子 benchmark 里的 3 倍**——这就是第 08 篇 Amdahl 提醒的工程现实。但在万卡规模的推理集群上,10%+ 的 decode 提速 = 实打实的机器成本,这正是 vLLM/SGLang 把这三件套全部 kernel 化的原因;
- 想要更大的提升,得动大头:attention(第 17/18 篇)和 Linear 的字节数(第 21 篇量化)。

---

## §5 练习 + 面试考点

### 动手练习

1. 把例 1 改成 **GPT-J/Neox interleaved 约定**(相邻偶奇维配对而非前后半),用 transformers 的 `rotate_half` 源码对拍,体会"约定不一致 = 静默错误"。
2. vLLM 的 `silu_and_mul` 输入是拼接的 `[T, 2*FFN]`(gate/up 连续存放)。改写例 2 接受拼接布局,想想为什么 vLLM 选择拼接(提示:少一次 Linear 调用,QKV 同理)。

### 面试高频考点

- **Q:RoPE 的原理?为什么主流模型都用它?**
  A:把 q/k 按维度两两配对旋转,角度 = 位置 × 频率;内积只依赖相对位置,天然相对编码;不加参数、可外推(配合 NTK/YaRN 缩放支持超长上下文);kernel 实现是纯逐元素 + 查表,代价极低。
- **Q:decode 阶段除了大矩阵乘,还有哪些可优化的算子?怎么优化?**
  A:RoPE、激活门控、残差+Norm、KV 写入等小算子,全部 memory-bound。手段:融合(减少读写与启动)、in-place(RoPE 原地旋转)、多输出(add+norm 同时产出两个张量)。整 block 收益 10%~40%,decode 形状下更明显。
- **Q:为什么 vLLM 把 silu 和 mul 做成一个算子、QKV 做成一个 Linear?**
  A:减少 kernel 启动与中间张量;两个 Linear 合一还能提高 GEMM 的 N 维大小,改善 Tensor Core 利用率(尤其 decode 的小 M 场景)。原则:能合的内存操作都合。
- **Q:单算子 benchmark 提速 3x,端到端只有 15%,你怎么向老板解释/怎么继续?**
  A:Amdahl:该算子端到端占比小。继续:profile 整条链路找最大占比项(通常 attention 与 GEMV);或者把优化"打包"(如 Liger 把整个 MLP 融成一个反向友好的大 kernel)提高可优化占比。
