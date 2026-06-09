# 13 · 反向传播 kernel:autograd.Function、atomic 与 RMSNorm backward

> Learn Triton 系列 · 阶段 2(核心算子)第 5 篇
> 前置:第 12 篇(RMSNorm 前向)、第 07 篇(归约)
> 运行环境:Google Colab T4 GPU

到目前为止我们的 kernel 都只管前向——推理够用了,但要让自定义算子进**训练**流程,必须把它接入 PyTorch 的自动求导:前向保存必要的中间量,反向用 kernel 算梯度,再用 `torch.autograd.Function` 把两者缝进计算图。本篇以 RMSNorm 为例走完全流程,并引入一个新原语:**atomic 操作**——跨 program 累加梯度的标准工具,也是"性能 vs 确定性"权衡的经典案例。

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

### 自定义算子进训练图的三件事

1. **数学**:手推(或查论文)反向公式。RMSNorm $y = \hat{x} \cdot w$,$\hat{x} = x \cdot r$,$r = (\overline{x^2} + \epsilon)^{-1/2}$:

$$\frac{\partial L}{\partial x} = r\left[ dy \odot w - \hat{x} \cdot \overline{(dy \odot w) \odot \hat{x}} \right], \qquad \frac{\partial L}{\partial w} = \sum_{\text{rows}} dy \odot \hat{x}$$

   ($\overline{\cdot}$ 表示行内均值。)注意结构:**dx 是行内归约(每行独立)**,**dw 是跨行归约(所有行贡献累加)**——两种归约,两种 kernel 策略。

2. **保存策略**:反向需要 $x$ 和 $r$。$x$ 本来就在;$r$ 每行一个标量,前向顺手存下(代价 4M 字节)比反向重算一遍划算。"存什么 vs 重算什么"是反向 kernel 设计的核心决策(第 24 篇 checkpointing 是它的宏观版)。

3. **封装**:`torch.autograd.Function` 的 `forward`(用 `ctx.save_for_backward` 存中间量)与 `backward`(返回与输入一一对应的梯度)。

### atomic:跨 program 的无序累加

dw 需要把 M 行的贡献加到同一个长度 N 的缓冲上,而 program 间不能同步(第 04 篇)。三种方案:

- **`tl.atomic_add(ptr, val)`**:硬件原子加,任意 program 随时往同一地址累加。简单、快,但**浮点加法顺序不确定 → 结果不可逐位复现**;
- **两阶段**:每个 program 写自己的 partial 行,再用第二个 kernel(或 torch.sum)归约——确定性,多一轮读写;
- **重排并行维度**:让一个 program 负责 dw 的一列、内部循环遍历所有行——确定性,但访存模式变差。

工业实践:**默认 atomic(快),提供确定性开关切两阶段**(PyTorch 的 `torch.use_deterministic_algorithms` 就是这个开关的全局版)。

### 能做什么

- 任何前向 kernel 都可以配上反向并接入 autograd,训练端到端可用(本篇实验直接训一个网络);
- 反向 kernel 同样吃融合红利:dx 的"两次归约+逐元素"一个 kernel 完成,比 autograd 自动微分生成的算子链快数倍;
- `tl.atomic_add` 支持 fp32/int(fp16 原子加在新硬件/新版本上部分支持)——dw 缓冲用 fp32 既为精度也为兼容;
- `gradcheck`(fp64)或"与 autograd 参考实现对照"(fp32/fp16 + 容差)两套验证手段。

### 不能做什么

- atomic 路线**不保证逐位可复现**:同一数据两次训练 loss 曲线可能在小数后几位分叉。对复现性硬要求的场景必须用确定性方案并接受性能损失;
- `ctx.save_for_backward` 只能存 tensor;保存太多中间量会让激活显存爆炸(第 24 篇算这笔账);
- 反向公式错了 autograd 不会报错,只会默默训坏——**验证梯度是不可跳过的纪律**;
- Triton 对 fp64 支持有限且极慢(消费级 GPU fp64 算力是 fp32 的 1/32~1/64),`gradcheck` 大网络不现实,常用 fp32 对照参考实现替代。

---

## §2 递进式例子

### 例 1:最小完整闭环 —— 自定义 SiLU 的前向 + 反向

SiLU(`x·sigmoid(x)`,Llama MLP 的激活)反向:$\frac{dy}{dx} = \sigma(x)(1 + x(1 - \sigma(x)))$。

```python
@triton.jit
def silu_fwd_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(y_ptr + offs, (x * tl.sigmoid(x)).to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu_bwd_kernel(x_ptr, dy_ptr, dx_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask).to(tl.float32)
    s = tl.sigmoid(x)
    dx = dy * s * (1.0 + x * (1.0 - s))
    tl.store(dx_ptr + offs, dx.to(x_ptr.dtype.element_ty), mask=mask)


class TritonSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.empty_like(x)
        n = x.numel()
        silu_fwd_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK=1024)
        ctx.save_for_backward(x)              # 反向要用 x(重算 sigmoid 比存 sigmoid 省显存)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        n = x.numel()
        silu_bwd_kernel[(triton.cdiv(n, 1024),)](x, dy.contiguous(), dx, n, BLOCK=1024)
        return dx                              # 与 forward 的输入一一对应


# 验证:前向 + 反向都与 PyTorch autograd 一致
x1 = torch.randn(65536, device="cuda", requires_grad=True)
x2 = x1.detach().clone().requires_grad_(True)
g = torch.randn(65536, device="cuda")

TritonSiLU.apply(x1).backward(g)
torch.nn.functional.silu(x2).backward(g)
torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-6)
print("SiLU 前向+反向 与 autograd 一致 ✓")
```

### 例 2:RMSNorm 完整训练算子 —— 行内归约(dx)+ atomic 跨行归约(dw)

```python
@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, M, N, stride_m, eps,
                       BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(rstd_ptr + row, rstd)                       # 给反向存的每行标量
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_m + cols, (x * rstd * w).to(x_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(x_ptr, w_ptr, dy_ptr, rstd_ptr, dx_ptr, dw_ptr,
                       M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)

    xhat = x * rstd
    dyw = dy * w
    # dx = rstd * (dyw - xhat * mean(dyw * xhat))    —— 行内归约,每行独立
    c = tl.sum(dyw * xhat, axis=0) / N
    dx = rstd * (dyw - xhat * c)
    tl.store(dx_ptr + row * stride_m + cols, dx.to(x_ptr.dtype.element_ty), mask=mask)

    # dw = sum_rows(dy * xhat)   —— 跨行归约:atomic_add 到 fp32 缓冲
    tl.atomic_add(dw_ptr + cols, dy * xhat, mask=mask)


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps=1e-6):
        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        BLOCK_N = triton.next_power_of_2(N)
        rmsnorm_fwd_kernel[(M,)](x, w, y, rstd, M, N, x.stride(0), eps,
                                 BLOCK_N=BLOCK_N, num_warps=8)
        ctx.save_for_backward(x, w, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rstd = ctx.saved_tensors
        M, N = x.shape
        dx = torch.empty_like(x)
        dw = torch.zeros(N, device=x.device, dtype=torch.float32)   # atomic 目标必须先清零
        BLOCK_N = triton.next_power_of_2(N)
        rmsnorm_bwd_kernel[(M,)](x, w, dy.contiguous(), rstd, dx, dw,
                                 M, N, x.stride(0), BLOCK_N=BLOCK_N, num_warps=8)
        return dx, dw.to(w.dtype), None


# 与 autograd 参考实现对照(前向 + dx + dw)
def rmsnorm_ref(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


M, N = 2048, 4096
x1 = torch.randn(M, N, device="cuda", requires_grad=True)
w1 = torch.randn(N, device="cuda", requires_grad=True)
x2 = x1.detach().clone().requires_grad_(True)
w2 = w1.detach().clone().requires_grad_(True)
g = torch.randn(M, N, device="cuda")

TritonRMSNorm.apply(x1, w1).backward(g)
rmsnorm_ref(x2, w2).backward(g)
torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(w1.grad, w2.grad, rtol=1e-3, atol=1e-3)
print("RMSNorm dx(行内归约)与 dw(atomic 跨行归约)都与 autograd 一致 ✓")
```

### 例 3:atomic 的代价 —— 不确定性实测

```python
def dw_once():
    dw = torch.zeros(N, device="cuda", dtype=torch.float32)
    dx = torch.empty_like(x1)
    rstd = torch.empty(M, device="cuda", dtype=torch.float32)
    rmsnorm_fwd_kernel[(M,)](x1.detach(), w1.detach(), torch.empty_like(x1), rstd,
                             M, N, x1.stride(0), 1e-6, BLOCK_N=triton.next_power_of_2(N))
    rmsnorm_bwd_kernel[(M,)](x1.detach(), w1.detach(), g, rstd, dx, dw,
                             M, N, x1.stride(0), BLOCK_N=triton.next_power_of_2(N))
    return dw

runs = [dw_once() for _ in range(5)]
bitwise_equal = all(torch.equal(runs[0], r) for r in runs[1:])
max_diff = max((runs[0] - r).abs().max().item() for r in runs[1:])
print(f"5 次运行 dw 逐位相等? {bitwise_equal};最大差异 {max_diff:.2e}")
print("atomic_add 的浮点累加顺序由调度决定 -> 数值在容差内正确,但可能不可逐位复现。")
print("训练框架的 deterministic 模式会改用两阶段归约(慢一些)换取复现性。")
```

---

## §3 知识连接

**与前面篇章:**

- dx 的"两次行内归约+逐元素"延续第 07/12 篇模式 A;dw 的跨行累加正是第 04 篇"program 间不能同步"的工程答案;
- 例 1"重算 sigmoid 而不存 sigmoid"与第 06 篇 dropout 的"重算 mask"同一哲学;第 24 篇 activation checkpointing 把它推广到层级;
- 第 08 篇验证纪律的扩展:反向必须独立验证(dx、dw 分开断言),不能只测前向。

**与真实框架:**

- Liger-Kernel `src/liger_kernel/ops/rms_norm.py`:与例 2 结构逐段对应(其 dw 用的是"每 program 写 partial 行 + torch 收尾"的确定性两阶段方案,正是 §1 列的方案二——读它的源码作为练习答案);
- PyTorch `torch.use_deterministic_algorithms(True)` 会让 `index_add` 等 atomic 类算子改走慢速确定性路径,报错信息里那句 "does not have a deterministic implementation" 的根源就是本篇例 3;
- `torch.compile` 也能自动生成反向 kernel(AOTAutograd 先把前向反向 trace 出来再交给 Inductor)——手写反向的价值在于:控制保存策略(省显存)与融合粒度,第 23 篇 fused CE 是极致案例(根本不物化 logits 梯度)。

---

## §4 闭环对比实验:把自定义算子放进真实训练

两个一模一样的两层 MLP(Linear → RMSNorm → SiLU → Linear),一个用 PyTorch 原生算子,一个用本篇的 TritonRMSNorm + TritonSiLU。训 300 步回归任务,对比:**loss 曲线是否重合(正确性)+ 每步耗时(性能)**。

```python
import matplotlib.pyplot as plt
import time

class MLP(torch.nn.Module):
    def __init__(self, use_triton, d=2048):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, 4 * d, bias=False)
        self.fc2 = torch.nn.Linear(4 * d, d, bias=False)
        self.w = torch.nn.Parameter(torch.ones(4 * d))
        self.use_triton = use_triton

    def forward(self, x):
        h = self.fc1(x)
        if self.use_triton:
            h = TritonRMSNorm.apply(h, self.w)
            h = TritonSiLU.apply(h)
        else:
            h = rmsnorm_ref(h, self.w)
            h = torch.nn.functional.silu(h)
        return self.fc2(h)


torch.manual_seed(42)
data_x = torch.randn(4096, 2048, device="cuda")
data_y = torch.randn(4096, 2048, device="cuda")

losses, step_ms = {}, {}
for name, use_triton in [("PyTorch 原生", False), ("Triton 自定义算子", True)]:
    torch.manual_seed(0)                       # 相同初始化,曲线才可比
    model = MLP(use_triton).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    curve = []
    torch.cuda.synchronize(); t0 = time.time()
    for step in range(300):
        idx = torch.randint(0, 4096, (256,), device="cuda")
        loss = torch.nn.functional.mse_loss(model(data_x[idx]), data_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(loss.item())
    torch.cuda.synchronize()
    losses[name] = curve
    step_ms[name] = (time.time() - t0) / 300 * 1000
    print(f"{name:16s} 最终 loss={curve[-1]:.5f}  平均每步 {step_ms[name]:.2f} ms")

diff = max(abs(a - b) for a, b in zip(losses["PyTorch 原生"], losses["Triton 自定义算子"]))
print(f"\n两条 loss 曲线最大偏差: {diff:.2e}(浮点顺序差异级别 -> 训练行为一致)")

plt.figure(figsize=(9, 4))
for name, c in losses.items():
    plt.plot(c, label=name, alpha=0.8)
plt.xlabel("step"); plt.ylabel("MSE loss"); plt.title("Training with custom Triton ops")
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

### 实验结果解读

- **两条 loss 曲线几乎重合**(偏差在浮点噪声量级):自定义算子的前向、反向、与 optimizer 的交互全部正确——这是"kernel 进训练"的最终验收标准,比单点 assert 更有说服力;
- 每步耗时:此网络主体是大 GEMM(走 cuBLAS),Norm/激活占比有限,因此 Triton 版的整体收益是小幅的——再次印证第 08 篇的 Amdahl 提醒:**单算子加速 ≠ 等比例端到端加速**。Norm/激活类融合的收益在算子占比更高的场景(长序列、小模型、或像 Liger 那样把整段 MLP 融掉)才会放大;
- 你现在拥有了"写一个能训练的自定义算子"的完整技能闭环:数学 → 前向/反向 kernel → autograd 封装 → 梯度验证 → 端到端验证。

---

## §5 练习 + 面试考点

### 动手练习

1. 把例 2 的 dw 改成**确定性两阶段**:反向 kernel 把每行贡献写到 `partial[M, N]`(或按 program 分组的 `partial[G, N]`),再 `partial.sum(0)` 收尾。对比 atomic 版的耗时与显存,并验证两次运行逐位一致。
2. 给第 12 篇练习 1 的 fused residual+RMSNorm 补上反向(注意残差分支的梯度直通),用本篇例 4 的训练实验验证。

### 面试高频考点

- **Q:自定义 CUDA/Triton 算子怎么接入 PyTorch 训练?**
  A:`torch.autograd.Function`:forward 里调前向 kernel 并 `save_for_backward` 反向所需张量,backward 里调反向 kernel 返回对应梯度;再包一层 `nn.Module`。验证:gradcheck(小规模 fp64)或与 autograd 参考实现对照(fp32 容差)。
- **Q:为什么 GPU 训练常常不可复现?哪里来的随机性?**
  A:浮点加法无结合律 + 并行累加顺序不固定。典型源头:atomic 类算子(scatter_add、embedding 反向、本篇 dw)、cuDNN 非确定性算法、多卡 all-reduce 顺序。对策:deterministic 模式(性能代价)、固定 seed 只能管 RNG 管不了调度顺序。
- **Q:LayerNorm/RMSNorm 反向的结构?难点在哪?**
  A:dx 是行内两次归约的逐元素组合(每行独立,好并行);dw/db 是跨行归约(与并行维度正交),需要 atomic 或两阶段。难点:统计量保存策略、mask 边界、fp32 累加。
- **Q:前向的中间结果,什么该存、什么该重算?**
  A:权衡显存与计算:标量/每行统计量(rstd)存(便宜);大张量(sigmoid 输出、dropout mask)倾向重算(省显存带宽,GPU 算力便宜);极端形态是 activation checkpointing 整层重算。判断依据:重算成本 vs 存储的字节 × 两次读写带宽成本。
