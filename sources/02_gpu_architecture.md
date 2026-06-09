# 02 · GPU 体系结构入门:SM、warp、存储层次与 roofline 模型

> Learn Triton 系列 · 阶段 0(地基)第 2 篇
> 前置:第 01 篇(GPU 异步执行模型、`gpu_time_ms` 计时函数)
> 运行环境:Google Colab T4 GPU

第 01 篇我们把 GPU 当黑盒:提交 kernel → 等结果。本篇打开这个黑盒:**kernel 在 GPU 内部是怎么被成千上万个线程执行的?数据在哪几层存储之间流动?** 最后用 **roofline 模型** 回答性能工程的第一问:*这个算子的天花板是算力还是带宽?*

写 Triton kernel 时你做的每一个决策(块大小、数据复用方式)本质上都是在和本篇讲的硬件结构对话。

## 环境准备

```python
import sys

import torch

assert torch.cuda.is_available(), (
    "未检测到 GPU!请在 Colab 菜单:代码执行程序 -> 更改运行时类型 -> 选择 T4 GPU"
)

props = torch.cuda.get_device_properties(0)
print(f"PyTorch  : {torch.__version__}")
print(f"GPU      : {torch.cuda.get_device_name(0)}")
print(f"显存     : {props.total_memory / 1024**3:.1f} GB")
print(f"算力版本 : sm_{props.major}{props.minor}")
print(f"SM 数量  : {props.multi_processor_count}")


# 第 01 篇定义的计时函数,本系列通用
def gpu_time_ms(fn, warmup=5, repeat=20):
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

### GPU 的组织结构:一座"流水线工厂"

一块 GPU 由几十个 **SM(Streaming Multiprocessor,流式多处理器)** 组成——T4 有 40 个 SM。每个 SM 内部有:

- 一批 **CUDA Core**(标量计算单元,T4 每 SM 64 个 FP32 core)和 **Tensor Core**(矩阵乘专用单元,速度比 CUDA core 高一个数量级);
- **寄存器堆(Register File)**:每个线程的私有变量住在这里,访问零延迟;
- **Shared Memory / L1**(T4 每 SM 64KB):同一个线程块内共享的高速缓存,程序员可控;
- 若干 **warp scheduler**:GPU 以 **warp(32 个线程一组)** 为单位发射指令,同一 warp 的 32 个线程**同一时刻执行同一条指令**(SIMT 模型)。

SM 之外是全局共享的 **L2 cache**(T4:4MB)和 **显存**(T4:16GB GDDR6,带宽约 320 GB/s)。

### 存储层次:速度差三个数量级

| 层次 | 容量(T4) | 带宽量级 | 谁能访问 |
|------|-----------|----------|----------|
| 寄存器 | 每 SM 256KB | ~每周期直达 | 单个线程私有 |
| Shared Memory / L1 | 每 SM 64KB | ~10 TB/s 级 | 同一线程块 |
| L2 Cache | 4MB | ~2 TB/s 级 | 全 GPU |
| 显存(GDDR6/HBM) | 16GB | **320 GB/s** | 全 GPU + 主机 |

**性能优化的核心就一句话:让数据尽量在金字塔上层被反复使用,少碰显存。** FlashAttention(第 14 篇)的全部精髓,就是把 attention 矩阵留在 shared memory 里不写回显存。

### memory-bound vs compute-bound

一个 kernel 的耗时下限由两个量决定:

- 计算时间下限 = 总浮点运算次数 FLOPs ÷ 峰值算力;
- 访存时间下限 = 总显存读写字节数 Bytes ÷ 显存带宽。

谁大谁说了算。衡量指标是 **算术强度(Arithmetic Intensity, AI)= FLOPs / Bytes**:

- `AI < 峰值算力/带宽`(T4 FP16 约 65T/320G ≈ 200)→ **memory-bound**(带宽是天花板):向量加法、ReLU、softmax、LayerNorm、**大模型 decode 阶段**几乎全部如此;
- `AI > 该比值` → **compute-bound**(算力是天花板):大矩阵乘、prefill 阶段的 attention。

**roofline 模型**把这两条天花板画成一条折线:横轴 AI、纵轴可达性能,左边斜线是带宽墙,右边平线是算力墙。

### 能做什么 / 不能做什么(这套心智模型的边界)

能做:

- 在写代码之前**预判**一个算子的性能上限,以及"优化还有没有空间";
- 解释为什么算子融合有效(提高 AI)、为什么 decode 慢(AI 极低);
- 指导 Triton kernel 的块大小、数据复用设计。

不能做:

- roofline 是**上限模型**,不解释"为什么没达到上限"(可能是访存不合并、occupancy 不足、bank conflict 等,后续篇章逐个展开);
- 它假设计算和访存完美重叠,真实 kernel 未必;
- 不建模 kernel 启动开销(第 01 篇实验 B)、CPU 端瓶颈和多 GPU 通信。

---

## §2 递进式例子

### 例 1:算出 T4 的两条"屋顶"——理论峰值

```python
# T4 (sm_75) 官方规格,作为 roofline 的两条屋顶
# 来源:NVIDIA Turing 架构白皮书 / T4 产品页
SPEC = {
    "fp32_tflops": 8.1,     # CUDA core: 2560 core * 2 FLOP * ~1.59GHz
    "fp16_tc_tflops": 65.0, # Tensor Core FP16
    "mem_bw_gbs": 320.0,    # GDDR6 显存带宽
    "l2_mb": 4,
    "sm_count": 40,
    "shared_kb_per_sm": 64,
}
ridge_fp16 = SPEC["fp16_tc_tflops"] * 1e12 / (SPEC["mem_bw_gbs"] * 1e9)
ridge_fp32 = SPEC["fp32_tflops"] * 1e12 / (SPEC["mem_bw_gbs"] * 1e9)
print(f"FP32 拐点(ridge point): AI = {ridge_fp32:.0f} FLOP/Byte")
print(f"FP16 TensorCore 拐点  : AI = {ridge_fp16:.0f} FLOP/Byte")
print("AI 低于拐点 -> memory-bound;高于拐点 -> compute-bound")
```

### 例 2:实测显存带宽——你能"吃到"理论值的几成?

用一个纯搬运操作(逐元素乘 2:读 4 字节、写 4 字节)测出**有效带宽**。

```python
n = 256 * 1024 * 1024 // 4  # 256MB 的 float32
x = torch.randn(n, device="cuda")

ms = gpu_time_ms(lambda: x.mul(2.0))
bytes_moved = 2 * x.numel() * x.element_size()  # 读一遍 + 写一遍
bw = bytes_moved / (ms / 1000) / 1e9
print(f"实测有效带宽: {bw:.1f} GB/s (理论 {SPEC['mem_bw_gbs']:.0f} GB/s, 达成率 {bw / SPEC['mem_bw_gbs'] * 100:.0f}%)")
MEASURED_BW = bw  # 后面 roofline 用实测值
```

### 例 3:实测算力——CUDA core vs Tensor Core

```python
n = 4096
a32 = torch.randn(n, n, device="cuda", dtype=torch.float32)
a16 = a32.half()

flops = 2 * n**3
ms32 = gpu_time_ms(lambda: a32 @ a32)
ms16 = gpu_time_ms(lambda: a16 @ a16)

tf32 = flops / (ms32 / 1000) / 1e12
tf16 = flops / (ms16 / 1000) / 1e12
print(f"FP32 matmul: {tf32:5.2f} TFLOPS (理论 {SPEC['fp32_tflops']}, 达成 {tf32/SPEC['fp32_tflops']*100:.0f}%)")
print(f"FP16 matmul: {tf16:5.2f} TFLOPS (理论 {SPEC['fp16_tc_tflops']}, 达成 {tf16/SPEC['fp16_tc_tflops']*100:.0f}%)")
print(f"Tensor Core 带来的实测加速: {ms32 / ms16:.1f}x")
MEASURED_FP16_TFLOPS = tf16
```

### 例 4:手算几个常见算子的算术强度

```python
def ai_report(name, flops, bytes_):
    ai = flops / bytes_
    bound = "memory-bound" if ai < MEASURED_FP16_TFLOPS * 1e12 / (MEASURED_BW * 1e9) else "compute-bound"
    print(f"{name:32s} AI = {ai:10.2f} FLOP/B  -> {bound}")
    return ai


N = 4096 * 4096  # 元素数
# 向量加法 c=a+b (fp32): 每元素 1 FLOP, 读 8B 写 4B
ai_report("vector add (fp32)", N, N * 12)
# ReLU (fp16): 读 2B 写 2B, 算 1 次比较
ai_report("ReLU (fp16)", N, N * 4)
# 矩阵乘 MxKxN=4096 (fp16): 2MNK FLOP, 读写 2(MK+KN+MN) 字节
M = K = Nn = 4096
ai_report("matmul 4096^3 (fp16)", 2 * M * K * Nn, 2 * (M * K + K * Nn + M * Nn))
# decode 阶段的 GEMV: batch=1 时 M=1
M = 1
ai_report("GEMV 1x4096x4096 (decode, fp16)", 2 * M * K * Nn, 2 * (M * K + K * Nn + M * Nn))
```

最后一行就是**大模型 decode 慢的本质**:batch=1 的矩阵向量乘 AI≈1,深陷 memory-bound 区,GPU 算力利用率不到 1%——这也是 continuous batching(第 19 篇)要解决的问题。

---

## §3 知识连接

**与第 01 篇的联系:**

- 第 01 篇说"kernel 启动有固定开销";本篇补上另一半:启动之后,性能由 SM 的并行度和存储层次决定;
- 实验 B 里"小 kernel 慢"还有个本篇视角的原因:任务太小填不满 40 个 SM(occupancy 不足)。

**与 Triton 的联系(第 03 篇起展开):**

- Triton 的一个 **program(程序实例)** 大致对应一个 CUDA 线程块,会被调度到某个 SM 上执行;
- 你在 Triton 里 `tl.load` 进来的数据块,编译器会安排进寄存器和 shared memory——块大小选多大,本质是在"复用率"和"每 SM 能同时跑几个块"之间权衡(第 11 篇 autotune 详谈);
- warp 和 SIMT 的概念解释了为什么 Triton 要求块内做"规整的"向量化操作。

**与真实框架的联系:**

- NVIDIA Nsight Compute 的报告里直接给出每个 kernel 的 roofline 图与 memory/compute 利用率,工业界调优第一步就是看它;
- vLLM/SGLang 的核心优化(PagedAttention、continuous batching)全部围绕"decode 是 memory-bound 且打不满 GPU"这一事实展开;
- FlashAttention 论文的 IO 复杂度分析,就是 roofline 思想在 attention 上的应用。

---

## §4 闭环对比实验:绘制 T4 的 roofline,并标出真实算子的位置

把例 2/例 3 的实测屋顶画出来,再实测 4 个算子,看它们落在图上哪里、离天花板多远。

```python
import matplotlib.pyplot as plt
import numpy as np

# ---- 实测 4 个算子:(名字, 可调用, FLOPs, Bytes) ----
n_vec = 64 * 1024 * 1024
va = torch.randn(n_vec, device="cuda")
vb = torch.randn(n_vec, device="cuda")

m4 = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
m1 = torch.randn(512, 512, device="cuda", dtype=torch.float16)
gemv_w = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
gemv_x = torch.randn(1, 4096, device="cuda", dtype=torch.float16)

ops = [
    ("vector add",        lambda: va + vb,        n_vec,            n_vec * 12),
    ("GEMV (decode-like)", lambda: gemv_x @ gemv_w, 2 * 4096 * 4096, 2 * (4096 + 4096 * 4096 + 4096)),
    ("matmul 512 fp16",   lambda: m1 @ m1,        2 * 512**3,       2 * 3 * 512**2),
    ("matmul 4096 fp16",  lambda: m4 @ m4,        2 * 4096**3,      2 * 3 * 4096**2),
]

points = []
for name, fn, flops, bytes_ in ops:
    ms = gpu_time_ms(fn)
    perf = flops / (ms / 1000) / 1e12          # 实测 TFLOPS
    ai = flops / bytes_                         # 算术强度
    points.append((name, ai, perf))
    print(f"{name:20s} AI={ai:9.2f}  实测 {perf:7.3f} TFLOPS  耗时 {ms:.3f} ms")

# ---- 画 roofline ----
ai_axis = np.logspace(-2, 4, 200)
roof = np.minimum(MEASURED_FP16_TFLOPS, MEASURED_BW / 1000 * ai_axis)  # TFLOPS

plt.figure(figsize=(9, 5))
plt.loglog(ai_axis, roof, "k-", lw=2, label=f"roofline (BW={MEASURED_BW:.0f}GB/s, peak={MEASURED_FP16_TFLOPS:.1f}TF)")
for name, ai, perf in points:
    plt.plot(ai, perf, "o", markersize=9)
    plt.annotate(name, (ai, perf), textcoords="offset points", xytext=(8, 5))
plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
plt.ylabel("Performance (TFLOPS)")
plt.title("T4 Roofline with measured operators")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()
```

### 实验结果解读

- **vector add** 和 **GEMV** 紧贴左侧斜线:它们已经把带宽吃满了,**继续优化计算毫无意义**,唯一出路是减少字节数(融合、量化、用更低精度);
- **matmul 4096** 靠近右侧平顶:compute-bound,优化方向是让 Tensor Core 持续吃饱(tiling、流水线,第 10/11 篇);
- **matmul 512** 离两条屋顶都远:规模太小,填不满 SM,启动/调度开销占比高——印证第 01 篇实验 A;
- 一张图建立直觉:**先算 AI,再决定优化策略**。这是面试中性能分析题的标准开场。

---

## §5 练习 + 面试考点

### 动手练习

1. 把例 2 的搬运操作分别换成 fp16(`x.half().mul(2.0)`)与 `x.clone()`,重新实测带宽,解释为什么 fp16 版本"耗时减半但带宽不变"。
2. 给 GEMV 实验增加 batch 维度(M = 1, 4, 16, 64, 256),画出"实测 TFLOPS vs batch"曲线,找到从 memory-bound 转为 compute-bound 的大致批量——这正是推理框架攒批(batching)的理论依据。

### 面试高频考点

- **Q:怎么判断一个算子是 memory-bound 还是 compute-bound?**
  A:算术强度 AI = FLOPs/Bytes,与硬件拐点(峰值算力/显存带宽)比较;低于拐点为 memory-bound。也可用 Nsight Compute 直接看两类利用率。
- **Q:大模型推理 decode 阶段为什么 GPU 利用率极低?**
  A:decode 每步 batch 小、每个权重只用一次,GEMV 的 AI≈1,远低于拐点(T4 约 200),时间全花在读权重上;所以优化方向是攒大 batch(continuous batching)、压缩权重字节数(量化)、减少重复读(KV cache 复用)。
- **Q:GPU 存储层次从快到慢说一遍?各自大概什么量级?**
  A:寄存器(每线程私有)→ shared memory/L1(每 SM,几十 KB,~10TB/s 级)→ L2(几 MB)→ 显存(GB 级,数百 GB/s)→ 主机内存(经 PCIe,几十 GB/s)。差距各约一个数量级,优化即"把热数据往上搬"。
- **Q:warp 是什么?为什么分支发散(divergence)伤性能?**
  A:warp 是 32 线程的锁步执行单位;同 warp 内走不同分支时硬件串行执行两条路径,利用率减半甚至更糟。Triton 的块式编程鼓励规整控制流来规避。
