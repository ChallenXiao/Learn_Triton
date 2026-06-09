# 10 · GEMM v1:tiling、tl.dot 与累加器

> Learn Triton 系列 · 阶段 2(核心算子)第 2 篇
> 前置:第 02 篇(Tensor Core、compute-bound)、第 04 篇(二维 grid 与广播索引)、第 05 篇(stride)
> 运行环境:Google Colab T4 GPU

矩阵乘(GEMM)是深度学习 90% 算力的去向:Linear 层、attention 的 QK^T 与 PV、LoRA、MoE 全是它。它也是面试的硬通货:**"手写一个分块矩阵乘"是 AI infra 岗的高频白板题**。本篇用 Triton 写出第一版 GEMM,理解 tiling(分块)为什么是一切高性能 GEMM 的骨架;下一篇再把它调到逼近 cuBLAS。

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

### 为什么 GEMM 能跑满算力:数据复用

$C = AB$($M\times K$ 乘 $K \times N$)有 $2MNK$ 次浮点运算,但只有 $MK + KN + MN$ 个数据——**每个数据平均被用 $O(N)$ 次**。第 02 篇说过它的算术强度随规模线性增长,是少数能冲到 roofline 平顶的算子。但前提是:**复用必须发生在片上**(寄存器/shared memory),而不是反复去显存重读。

### tiling:三层分块的心智模型

```text
把 C 切成 BLOCK_M x BLOCK_N 的瓦片,每个 program 负责一片:

    for k in range(0, K, BLOCK_K):                 # 沿 K 维滑动
        a_tile = A[rm, k:k+BLOCK_K]                # [BM, BK] 读入片上
        b_tile = B[k:k+BLOCK_K, rn]                # [BK, BN] 读入片上
        acc += a_tile @ b_tile                     # tl.dot -> Tensor Core
    C[rm, rn] = acc                                # 累加器只在最后写出一次
```

复用账:`a_tile` 的每个元素参与 BLOCK_N 列的计算,`b_tile` 的每个元素参与 BLOCK_M 行——**块越大,每字节显存流量摊到的计算越多**(AI ≈ 块尺寸的一半)。代价是片上资源占用越大,这是第 11 篇调参的核心矛盾。

### `tl.dot`:通往 Tensor Core 的门

`tl.dot(a, b)` 对两个二维块做矩阵乘,编译器把它映射到 Tensor Core 的 `mma` 指令。硬约束:

- 两个操作数的各维度**至少 16**(Tensor Core 的最小瓦片);
- 输入用 fp16/bf16(T4 只支持 fp16 走 Tensor Core),**累加器必须 fp32**(`tl.zeros(..., dtype=tl.float32)`)——精度与硬件行为的双重要求;
- 这是 Triton 里**唯一**触发 Tensor Core 的途径;写成逐元素乘加循环,性能差一个数量级。

### 能做什么

- 50 行写出正确且像样的 GEMM(本篇),调优后达到 cuBLAS 的 80%~95%(下一篇);
- 任意形状、任意 stride(转置布局直接传 stride 即可,不需要先 `.contiguous()`);
- 在 K 循环前后自由插入代码——这是 cuBLAS 做不到的:**prologue 融合**(边载入边反量化,第 21 篇)和 **epilogue 融合**(加 bias、激活,第 12 篇);
- attention 的核心(两个 matmul 夹一个 softmax)就是本篇骨架 + 第 09 篇 online 统计的组合(第 14 篇)。

### 不能做什么

- **常规大 GEMM 别指望打败 cuBLAS**:cuBLAS/CUTLASS 有汇编级流水线、为每个架构和形状预调的配置表,Triton 一般只能逼近;手写的价值在"融合"与"特殊形状/布局/精度",不在裸 GEMM 本身;
- `tl.dot` 维度 < 16 直接不可用(超小块场景如某些 decode GEMV 要换写法);
- fp64 GEMM、确定性逐位复现 cuBLAS 结果,不在能力范围(浮点求和顺序不同);
- T4(sm_75)上 bf16 的 `tl.dot` 不可用——本系列统一用 fp16。

---

## §2 递进式例子

### 例 1:第一版 GEMM kernel —— 50 行的完整实现

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # 本片负责的行
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # 本片负责的列
    rk = tl.arange(0, BLOCK_K)                        # K 维滑动窗口内偏移

    # A 的 [BM, BK] 块指针、B 的 [BK, BN] 块指针(第 04 篇的广播 idiom)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # fp32 累加器(硬规矩)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)                                # Tensor Core 在此
        a_ptrs += BLOCK_K * stride_ak                      # 窗口右移
        b_ptrs += BLOCK_K * stride_bk                      # 窗口下移

    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
             acc.to(tl.float16), mask=c_mask)


def triton_matmul(a, b, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == torch.float16
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                        num_warps=num_warps, num_stages=num_stages)
    return c


a = torch.randn(1000, 700, device="cuda", dtype=torch.float16)  # 三个维度都不整除
b = torch.randn(700, 900, device="cuda", dtype=torch.float16)
torch.testing.assert_close(triton_matmul(a, b), a @ b, rtol=1e-2, atol=1e-2)
print("GEMM 正确 ✓ (1000x700 @ 700x900, 全维度非整除)")
```

### 例 2:精度账 —— fp32 累加器到底重不重要

fp16 只有 10 位尾数,K 个数连加的误差随 K 增长。以 fp64 全精度为基准,对比"fp16 输入 + fp32 累加"(我们和 cuBLAS 的做法)与"全程 fp16 累加"(模拟:对 fp16 中间结果反复舍入)。

```python
torch.manual_seed(0)
K_list = [256, 1024, 4096, 16384]
M = N = 64
print(f"{'K':>6} | {'fp32 累加误差':>14} | {'fp16 累加误差':>14}")
for K in K_list:
    a64 = torch.randn(M, K, device="cuda", dtype=torch.float64)
    b64 = torch.randn(K, N, device="cuda", dtype=torch.float64)
    ref = a64 @ b64
    # fp16 输入 + fp32 累加(triton_matmul 的行为)
    out_acc32 = triton_matmul(a64.half(), b64.half()).double()
    # 全程 fp16:分段乘加后立刻舍回 fp16(模拟 fp16 累加器)
    out_acc16 = torch.zeros(M, N, device="cuda", dtype=torch.float16)
    for k0 in range(0, K, 256):
        out_acc16 = (out_acc16 + (a64[:, k0:k0+256].half() @ b64[k0:k0+256, :].half())).half()
    err32 = (out_acc32 - ref).abs().max().item()
    err16 = (out_acc16.double() - ref).abs().max().item()
    print(f"{K:>6} | {err32:>14.4f} | {err16:>14.4f}")
print("\nfp32 累加的误差几乎不随 K 增长;fp16 累加误差随 K 放大数倍 —— 这就是'累加器必须 fp32'的原因")
```

### 例 3:stride 的红利 —— 转置布局免费支持

实际模型里权重经常以转置形式存储(`nn.Linear` 存的是 `[out, in]`,前向要算 `x @ W^T`)。例 1 的 kernel 不用改一行,传转置后的 stride 即可。

```python
x = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
w = torch.randn(2048, 1024, device="cuda", dtype=torch.float16)   # nn.Linear 风格 [out, in]

# x @ w.T:把 w.T 当作 B 传入,b.stride 自动给出 (1, 1024)
out = triton_matmul(x, w.t())
torch.testing.assert_close(out, x @ w.t(), rtol=1e-2, atol=1e-2)
print("x @ W^T 正确 ✓ —— 无需物化转置,stride 直接描述布局(第 05 篇的回报)")
print("注意:转置侧的访存合并性会变差,性能影响在下一篇 autotune 中可被部分缓解")
```

---

## §3 知识连接

**与前面篇章:**

- 第 04 篇例 1 的二维分块加法 = 本篇去掉 K 循环的退化版;`rm[:, None] * stride + rn[None, :]` 的 idiom 原样复用;
- 第 02 篇 roofline:本篇 kernel 的 AI ≈ BLOCK 尺寸量级(64 块 ≈ 32 FLOP/B),已越过 T4 的 fp16 拐点(~200?未必——这正是块大小要继续调大的动机,第 11 篇);
- 第 05 篇合并访存:`rk[None, :] * stride_ak`(stride_ak=1 时)保证 A 块行内连续读。

**与 CUDA/CUTLASS 对照:**

- CUDA 手写 GEMM 的标准套路(shared memory 双缓冲、寄存器分块、`mma.sync`)在 Triton 里被压缩成:`tl.dot` + `num_stages`(软件流水线级数,编译器自动生成双/多缓冲);
- CUTLASS 把这套结构模板化成 C++ 库;Triton 把它语言化——两者的 tiling 层次结构完全同构(threadblock tile / warp tile / instruction tile)。

**与真实框架:**

- `torch.matmul` 在 NVIDIA 上调 cuBLAS(cublasLtMatmul);本篇实验直接以它为标杆;
- vLLM 的 LoRA kernel(punica 系列)、量化 GEMM(`vllm/model_executor/layers/quantization/` 下的 Triton kernel)、Liger 的各种 fused linear,骨架都是本篇这 50 行的变体;
- Triton 官方教程 03-matrix-multiplication 是本篇的对照读物,其中 L2 swizzle 部分我们留到第 11 篇。

---

## §4 闭环对比实验:v1 vs cuBLAS —— 差距有多大,差在哪

方阵规模 512→4096,对比本篇固定配置(64×64×32)与 cuBLAS 的 TFLOPS。

```python
import matplotlib.pyplot as plt

sizes = [512, 1024, 2048, 4096]
tf_triton, tf_cublas = [], []

for n in sizes:
    A = torch.randn(n, n, device="cuda", dtype=torch.float16)
    B = torch.randn(n, n, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(triton_matmul(A, B), A @ B, rtol=1e-2, atol=1e-2)
    flops = 2 * n**3
    ms_t = triton.testing.do_bench(lambda: triton_matmul(A, B), return_mode="median")
    ms_c = triton.testing.do_bench(lambda: A @ B, return_mode="median")
    tf_triton.append(flops / (ms_t / 1000) / 1e12)
    tf_cublas.append(flops / (ms_c / 1000) / 1e12)

print(f"{'规模':>6} | {'Triton v1':>10} | {'cuBLAS':>8} | {'达成率':>7}")
for n, tt, tc in zip(sizes, tf_triton, tf_cublas):
    print(f"{n:>6} | {tt:>8.2f}TF | {tc:>6.2f}TF | {tt / tc * 100:>6.1f}%")

plt.figure(figsize=(8, 4.5))
plt.plot(sizes, tf_cublas, "s-", label="cuBLAS (torch.matmul)")
plt.plot(sizes, tf_triton, "o-", label="Triton v1 (fixed 64x64x32)")
plt.axhline(65, color="gray", ls="--", label="T4 fp16 peak 65 TFLOPS")
plt.xlabel("matrix size N"); plt.ylabel("TFLOPS")
plt.title("GEMM v1 vs cuBLAS on T4")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 实验结果解读

- v1 通常达到 cuBLAS 的 **50%~75%**:正确的骨架给了我们第一桶金(Tensor Core + 片上复用),但固定的 64×64×32 配置对不同规模都不是最优;
- 差距来源(第 11 篇逐项收复):① 块太小,AI 不够高、Tensor Core 喂不饱;② `num_stages=2` 的流水线太浅,访存延迟没被计算盖住;③ program 的执行顺序没有考虑 L2 复用(swizzle);
- cuBLAS 自己也没到 65 TFLOPS 平顶——T4 的功耗墙和实际可持续频率决定了真实上限,这是"理论峰值 vs 可达峰值"的现实一课。

---

## §5 练习 + 面试考点

### 动手练习

1. 把 BLOCK_M/N/K 手动改成 (32,32,32) 和 (128,128,32) 重跑实验,观察"块大小 ↔ 性能"并尝试解释两端为什么都不好(片上资源 vs 复用率)。
2. 给 kernel 加一个 epilogue:`C = relu(A@B + bias)`(bias 沿 N 广播)。验证正确性,对比"先 matmul 再 eager 加 bias+relu"的两 kernel 方案——你已经在写第 12 篇的内容了。

### 面试高频考点

- **Q:白板写分块矩阵乘的伪代码,并解释为什么分块能加速?**
  A:三重分块(输出瓦片 × K 滑窗 × 片上累加),见 §1 心智模型。加速本质:把 $O(MNK)$ 的计算摊到 $O(MK+KN+MN)$ 的显存流量上,块内数据在寄存器/shared memory 复用 BLOCK 次,显存流量降为朴素实现的 1/BLOCK。
- **Q:为什么累加器要用 fp32,输入却可以是 fp16?**
  A:乘积单项的动态范围 fp16 尚可承受,但 K 项连加会累积舍入误差且可能溢出;Tensor Core 硬件本身就支持 fp16 乘 + fp32 加。例 2 的实测:fp16 累加误差随 K 放大数倍。
- **Q:你写的 Triton GEMM 比 cuBLAS 慢 30%,优化思路?**
  A:① autotune 块大小/warps/stages 匹配形状与硬件;② 加深 num_stages 让 load 与 dot 重叠;③ program 重排(swizzle)提高 L2 命中;④ 检查两个操作数的访存合并性;⑤ 若仍差,这个形状可能就该用 cuBLAS——手写的真正战场是融合与特殊精度(下一篇与第 12/21 篇展开)。
- **Q:什么时候 GEMM 反而是 memory-bound?**
  A:任一维度很小时(decode 的 M=1 GEMV、LoRA 的小 rank、MoE 的小 expert 批),AI 退化到 O(1),权重读取主导——此时优化目标从"喂饱 Tensor Core"变成"减少字节"(量化、第 21 篇)和"攒批"(第 19 篇)。
