# 08 · 工程方法论:benchmark、正确性验证与调试

> Learn Triton 系列 · 阶段 1(编程模型)第 5 篇(阶段收尾)
> 前置:第 01 篇(Event 计时)、第 03-07 篇(已写过若干 kernel)
> 运行环境:Google Colab T4 GPU

前面几篇我们一直用自制的 `gpu_time_ms` 计时。它能用,但有一个隐蔽的坑会让你**高估性能 50% 以上**(本篇实验亲手复现)。性能工程的第一信条是:**测不准,一切优化都是自欺**。本篇建立本系列后续 21 篇统一使用的测量、验证、调试三件套——这也是你将来在公司里给 kernel 出 benchmark 报告时的规范。

## 环境准备

```python
import torch

assert torch.cuda.is_available(), "请在 Colab 选择 GPU 运行时"

import triton
import triton.language as tl
import triton.testing

print(f"PyTorch {torch.__version__} | Triton {triton.__version__} | {torch.cuda.get_device_name(0)}")


def gpu_time_ms(fn, warmup=5, repeat=20):
    """前几篇用的朴素 Event 计时 —— 本篇将揭示它的问题。"""
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

### GPU benchmark 的四大陷阱

1. **不预热**:首次调用包含 JIT 编译(Triton kernel 第一次跑要编译几百毫秒)、cuDNN 算法选择、显存分配器建池。必须预热到稳态再测;
2. **不同步**:第 01 篇讲过的异步陷阱,测了个提交时间;
3. **L2 缓存热**(最隐蔽):反复对**同一块数据**计时,第二次起数据可能还在 4MB 的 L2 里,测出的"显存带宽"实际是 L2 带宽,虚高 30%~100%。真实负载里数据通常是冷的;
4. **时钟漂移**:GPU 会动态升降频(温度、功耗墙)。长时间 benchmark 中后段可能降频;Colab 共享环境还有邻居干扰。对策:取中位数/分位数而非平均,关注波动幅度。

### 标准工具:`triton.testing.do_bench`

```py
ms = triton.testing.do_bench(fn)                       # 返回均值
ms_med, ms_min, ms_max = triton.testing.do_bench(
    fn, quantiles=[0.5, 0.2, 0.8])                     # 中位数与分位数
```

它替你处理了全部四个坑:自动预热、自动同步、**每次迭代之间清空 L2 缓存**(向一个大缓冲写入,把缓存挤干净)、按时间预算自动决定重复次数并报告分位数。**本系列从本篇起统一用 do_bench。**

### 正确性验证的规范

- `torch.testing.assert_close(actual, expected, rtol, atol)`:默认容差按 dtype 自动选择(fp32: rtol=1.3e-6;fp16: rtol=1e-3),浮点并行计算**不应该也不可能**逐位相等(第 07 篇);
- 验证要覆盖**边界形状**:非 2 的幂、不被 BLOCK 整除、极小(1×1)、单行单列——mask 的 bug 只在边界暴露;
- 数值稳定性单独测:喂大数(1e4)、全负数、全相同值,看是否 NaN/Inf(第 09 篇展开)。

### 调试手段(按成本递增)

1. **`tl.device_print("label", value)`**:kernel 内打印,适合看个别 program 的中间值(输出量大,只在小 grid 上用);
2. **`TRITON_INTERPRET=1`**:解释器模式,kernel 在 CPU 上用 NumPy 模拟执行,可以加 Python 断点、打印任意变量。**必须在 `import triton` 之前设置环境变量**,所以 notebook 里要用"重启 + 开头设置"或子进程方式;
3. **二分缩小**:把 kernel 拆半,先只 load 后立即 store,验证索引;再逐步加回计算——索引错误占 Triton bug 的 80%;
4. **读编译产物**:`TRITON_KERNEL_DUMP=1` 导出各阶段 IR(第 28 篇用到)。

### 能做什么 / 不能做什么

能做:可复现的单 kernel 微基准、分位数稳定性报告、跨实现公平对比(同输入、同清缓存条件)。

不能做:

- **微基准 ≠ 端到端性能**:单 kernel 快 3 倍,端到端可能只快 3%(Amdahl 定律)——结论必须配 profile 占比说话(第 21/28 篇用 profiler 补全这块);
- do_bench 测不了多 kernel 流水线的重叠效应、CPU 调度间隙(需要 `torch.profiler` 时间线);
- Colab 上的绝对数值会漂移(共享卡、降频),**同一 cell 内对比的相对结论可信,跨会话的绝对值不可比**。

---

## §2 递进式例子

### 例 1:do_bench 基本用法与分位数

```python
x = torch.randn(16 * 1024 * 1024, device="cuda")
y = torch.randn_like(x)

fn = lambda: x + y

ms_mean = triton.testing.do_bench(fn)
med, lo, hi = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
print(f"均值 {ms_mean:.4f} ms | 中位数 {med:.4f} ms | 20%~80% 分位 [{lo:.4f}, {hi:.4f}] ms")
print(f"波动幅度 {(hi - lo) / med * 100:.1f}% —— 报告里永远带上波动,只给单个数字是不专业的")
```

### 例 2:标准化对比报告 —— 本系列后续篇章的模板

把"多实现 × 正确性 × 分位数 × 带宽/FLOPS 折算"封装成一个函数,后面每篇的 §4 实验都是它的实例。

```python
def bench_report(impls: dict, ref_name: str, bytes_moved=None, flops=None,
                 rtol=1e-4, atol=1e-4):
    """impls: {名字: 无参可调用};ref_name: 作为正确性基准的实现名。"""
    ref = impls[ref_name]()
    rows = []
    for name, fn in impls.items():
        torch.testing.assert_close(fn(), ref, rtol=rtol, atol=atol)  # 先验证再计时
        med, lo, hi = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
        extra = ""
        if bytes_moved:
            extra = f"{bytes_moved / (med / 1000) / 1e9:8.1f} GB/s"
        if flops:
            extra = f"{flops / (med / 1000) / 1e12:8.2f} TFLOPS"
        rows.append((name, med, lo, hi, extra))
    base = rows[0][1]
    print(f"{'实现':24s} {'中位数':>9} {'波动区间':>19} {'折算':>12} {'相对加速':>8}")
    for name, med, lo, hi, extra in rows:
        print(f"{name:24s} {med:8.4f}ms [{lo:7.4f},{hi:7.4f}]ms {extra:>12} {base / med:7.2f}x")


# 用第 07 篇的 softmax 当例子
@triton.jit
def softmax_kernel(x_ptr, out_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    v = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=float("-inf"))
    v = v - tl.max(v, axis=0)
    num = tl.exp(v)
    tl.store(out_ptr + row * stride_m + cols, num / tl.sum(num, axis=0), mask=mask)


def triton_softmax(t):
    M, N = t.shape
    out = torch.empty_like(t)
    softmax_kernel[(M,)](t, out, M, N, t.stride(0), BLOCK_N=triton.next_power_of_2(N), num_warps=8)
    return out


t = torch.randn(8192, 2048, device="cuda")
bench_report(
    {"PyTorch eager": lambda: torch.softmax(t, -1), "Triton": lambda: triton_softmax(t)},
    ref_name="PyTorch eager",
    bytes_moved=2 * t.numel() * 4,
)
```

### 例 3:kernel 内打印与解释器模式

```python
@triton.jit
def debug_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    if pid == 0:                       # 只让 0 号 program 打印,避免刷屏
        tl.device_print("pid0 partial sum = ", s)
    tl.store(out_ptr + pid, s)


small = torch.ones(100, device="cuda")
out = torch.empty(4, device="cuda")
debug_kernel[(4,)](small, out, 100, BLOCK=32)
torch.cuda.synchronize()   # device_print 的输出要等同步后才出现
print("各 program 部分和:", out.cpu().tolist(), "(0..2 号是 32, 3 号是 4 个 1)")

# 解释器模式说明(不在此 cell 运行):
# TRITON_INTERPRET=1 必须在 import triton 之前设置。notebook 里的用法:
#   1) Colab: 修改代码第一个 cell 为 import os; os.environ["TRITON_INTERPRET"]="1",
#      然后 重启运行时 再顺序执行 —— kernel 将以 NumPy 在 CPU 模拟运行,可下断点;
#   2) 或者把 kernel 放进独立 .py 文件,用 !TRITON_INTERPRET=1 python debug.py 跑子进程。
```

---

## §3 知识连接

**与前面篇章:**

- `gpu_time_ms`(第 01 篇)≈ do_bench 去掉"清 L2"与自动重复策略——本篇 §4 实验定量展示这一差异;前几篇的数据都是大张量(远超 4MB L2),所以结论依然成立,但小数据时必须换工具;
- 第 07 篇"浮点归约不可逐位相等"在这里落地成 assert_close 的容差规范。

**与真实框架:**

- `triton.testing` 还提供 `perf_report` / `Benchmark` 装饰器(官方教程每篇结尾的扫描曲线就是它画的),我们的 `bench_report` 是它的轻量版;
- vLLM 的 `benchmarks/kernels/` 目录、Liger-Kernel 的 `benchmark/` 目录里的脚本结构与本篇模板一致:正确性 → 分位数计时 → 折算指标 → 多形状扫描,读懂本篇即可读懂它们;
- PyTorch 官方的 `torch.utils.benchmark.Timer` 是 eager 侧的等价物(自动预热 + 自适应重复),跨框架对比时可作为第二信源。

---

## §4 闭环对比实验:朴素计时 vs do_bench —— L2 缓存怎么骗人

同一个逐元素 kernel,数据规模从"远小于 L2(4MB)"扫到"远大于 L2"。朴素计时反复加热同一块数据,小数据时测出来的是 L2 带宽;do_bench 每次迭代清空 L2,测的才是真实显存带宽。

```python
import matplotlib.pyplot as plt

@triton.jit
def mul2_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)


sizes_mb = [0.5, 1, 2, 4, 8, 16, 64, 256]
naive_bw, honest_bw = [], []

for mb in sizes_mb:
    n = int(mb * 1024 * 1024 / 4)
    xx = torch.randn(n, device="cuda")
    oo = torch.empty_like(xx)
    fn = lambda: mul2_kernel[(triton.cdiv(n, 1024),)](xx, oo, n, BLOCK=1024)
    bytes_moved = 2 * n * 4

    ms_naive = gpu_time_ms(fn)                                    # 朴素:数据在 L2 里越来越热
    ms_honest = triton.testing.do_bench(fn, return_mode="median")  # do_bench:每次清 L2
    naive_bw.append(bytes_moved / (ms_naive / 1000) / 1e9)
    honest_bw.append(bytes_moved / (ms_honest / 1000) / 1e9)

print(f"{'数据量':>8} | {'朴素计时 GB/s':>13} | {'do_bench GB/s':>13} | {'虚高':>6}")
for mb, nb, hb in zip(sizes_mb, naive_bw, honest_bw):
    print(f"{mb:>6.1f}MB | {nb:>13.1f} | {hb:>13.1f} | {nb / hb:>5.2f}x")

plt.figure(figsize=(9, 4.5))
plt.semilogx(sizes_mb, naive_bw, "o-", label="naive timing (L2-hot)")
plt.semilogx(sizes_mb, honest_bw, "s-", label="do_bench (L2 flushed)")
plt.axvline(4, color="red", ls=":", label="T4 L2 = 4MB")
plt.axhline(320, color="gray", ls="--", label="DRAM 320 GB/s")
plt.xlabel("working set (MB)")
plt.ylabel("apparent bandwidth (GB/s)")
plt.title("Why naive benchmarking lies: L2 cache effect")
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 实验结果解读

- 数据 ≤ 4MB 时,朴素计时报出的"带宽"**远超 320 GB/s 的物理显存带宽**——显然在测 L2,不是 DRAM;do_bench 曲线则始终诚实;
- 数据 ≫ L2 后两条曲线收敛:这就是为什么前几篇用大张量做实验时朴素计时没出事;
- 工程教训:**优化推理 decode kernel(working set 常常很小)时,用错计时工具会让你把假提升当成果汇报**。从下一篇起,本系列所有实验统一切换到 `do_bench`。

---

## §5 练习 + 面试考点

### 动手练习

1. 给 `bench_report` 加一列"峰值显存"(`torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_allocated()`),后面 FlashAttention 篇(第 14 篇)会需要它。
2. 故意制造一个索引 bug(例 2 的 softmax 把 `stride_m` 传成 `N-1`),用"二分缩小法"(先改成纯 copy kernel)定位问题,体会调试流程。

### 面试高频考点

- **Q:给一个 GPU kernel 做 benchmark,要注意什么?**
  A:① 预热排除 JIT/缓存建立;② 正确同步(Event 或 sync);③ 清 L2 防止缓存热虚高(do_bench 的做法:迭代间写大缓冲);④ 报中位数+分位数抗时钟漂移;⑤ 对比实验控制同输入、同条件;⑥ 微基准结论要配端到端 profile 占比。
- **Q:怎么验证自定义 kernel 的正确性?**
  A:与参考实现 `assert_close`(按 dtype 选容差,不要求逐位相等);覆盖边界形状(非整除、极小、单行);单独测数值稳定性(大数、特殊值);若有反向,用 `torch.autograd.gradcheck`(fp64)。
- **Q:Triton kernel 怎么调试?**
  A:`tl.device_print` 看中间值;`TRITON_INTERPRET=1` 用 CPU 解释执行可断点(注意要在 import 前设置);二分法先验证索引(load 后直接 store)再加回计算;`TRITON_KERNEL_DUMP` 看各级 IR。
- **Q:单 kernel 提速 3 倍,端到端只快 3%,为什么?**
  A:Amdahl 定律——该 kernel 端到端占比可能只有 5%。优化前必须先 profile 拿到时间占比,按占比 × 提速空间排优先级,而不是按"好写"排。
