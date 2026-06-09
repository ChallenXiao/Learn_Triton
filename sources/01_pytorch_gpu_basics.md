# 01 · PyTorch Tensor 与 GPU 执行模型速成

> Learn Triton 系列 · 阶段 0(地基)第 1 篇
> 前置要求:会写基础 Python。
> 运行环境:Google Colab(菜单:代码执行程序 → 更改运行时类型 → **T4 GPU**)

本篇是整个系列的地基:在写任何 Triton kernel 之前,你必须先理解 **PyTorch 是如何把计算交给 GPU 的**,以及**为什么"测得快"和"真的快"是两回事**。本系列后面 28 篇的所有性能对比实验,都建立在本篇的计时方法之上。

## 环境准备

每篇 notebook 的第一个 cell 都是这段环境检测代码:

```python
import sys

import torch

assert torch.cuda.is_available(), (
    "未检测到 GPU!请在 Colab 菜单:代码执行程序 -> 更改运行时类型 -> 选择 T4 GPU"
)

props = torch.cuda.get_device_properties(0)
print(f"Python   : {sys.version.split()[0]}")
print(f"PyTorch  : {torch.__version__}")
print(f"GPU      : {torch.cuda.get_device_name(0)}")
print(f"显存     : {props.total_memory / 1024**3:.1f} GB")
print(f"算力版本 : sm_{props.major}{props.minor} (Compute Capability {props.major}.{props.minor})")
print(f"SM 数量  : {props.multi_processor_count}")
```

---

## §1 是什么 & 能力边界

### 这个主题是什么

PyTorch 的核心对象是 **Tensor(张量)**:一块带形状(shape)、数据类型(dtype)、所在设备(device)信息的连续内存。当 Tensor 在 GPU 上时,对它的每一个操作(加法、矩阵乘……)都会变成一次 **CUDA kernel 启动(kernel launch)**——CPU 把"要做什么"提交到 GPU 的命令队列(CUDA Stream),然后**立刻返回,不等 GPU 算完**。这就是 **异步执行(asynchronous execution)** 模型。

理解这个模型,你才能回答三个本系列反复出现的问题:

1. 一段 GPU 代码的时间花在哪里?(CPU 提交开销 / GPU 计算 / 内存搬运)
2. 为什么 `time.time()` 测 GPU 代码经常测出错误结果?
3. 为什么"把很多小操作合并成一个大操作"会快?——这正是 Triton 算子融合的动机。

### 它的作用

- 提供 **device 抽象**:同一份代码,`tensor.to("cuda")` 即可切换到 GPU;
- 自动管理 GPU 显存(caching allocator)与 kernel 调度,你不需要写任何 CUDA;
- 它是本系列的"对照组":后面我们写的每一个 Triton kernel,都要和 PyTorch 原生实现比正确性、比速度。

### 能做什么(能力边界内)

- 用一行代码调用高度优化的算子(`torch.matmul` 背后是 NVIDIA cuBLAS 库,接近硬件极限);
- 异步提交大量 kernel,让 CPU 和 GPU 流水线并行工作;
- 用 `torch.cuda.Event` 精确测量 GPU 上的真实耗时;
- 用 `torch.cuda.memory_allocated()` 等接口观测显存。

### 不能做什么(能力边界外)

- **不能自动融合算子**(eager 模式下):`x.mul(2).add(3).relu()` 是 3 次独立的 kernel 启动、3 次完整的显存读写——这是性能浪费的主要来源,也是 Triton / torch.compile 存在的理由;
- **不能控制 kernel 内部行为**:线程如何划分、数据如何在片上缓存复用,PyTorch 用户层完全摸不到——想控制就要写 Triton/CUDA;
- **小操作的启动开销无法避免**:每次 kernel launch 有约 5~30 微秒的固定开销,操作太小时 GPU 大部分时间在"等活"而不是"干活"。

---

## §2 递进式例子

### 例 1:Tensor 的三要素 —— shape / dtype / device

```python
import torch

# 在 CPU 上创建,再搬到 GPU
x_cpu = torch.randn(4, 3)                    # 默认 float32、在 CPU
x_gpu = x_cpu.to("cuda")                     # 拷贝到 GPU(经过 PCIe 总线)
y_gpu = torch.randn(4, 3, device="cuda")     # 直接在 GPU 上创建(推荐,省一次拷贝)

print("x_cpu:", x_cpu.shape, x_cpu.dtype, x_cpu.device)
print("x_gpu:", x_gpu.shape, x_gpu.dtype, x_gpu.device)

# 不同 device 的 tensor 不能直接运算——这条报错你以后会经常见到
try:
    _ = x_cpu + y_gpu
except RuntimeError as e:
    print("报错(符合预期):", e)

# dtype 决定每个元素占多少字节:这直接决定显存占用和内存带宽消耗
for dt in [torch.float32, torch.float16, torch.bfloat16, torch.int8]:
    t = torch.zeros(1024, 1024, dtype=dt, device="cuda")
    print(f"{str(dt):20s} 每元素 {t.element_size()} 字节, 1024x1024 共 {t.numel() * t.element_size() / 1024**2:.1f} MB")
```

### 例 2:异步执行 —— 为什么 `time.time()` 会"测了个寂寞"

CPU 提交 kernel 后立刻返回。如果你直接用 `time.time()` 夹住一段 GPU 代码,测到的往往只是**提交命令的时间**,不是 GPU 真正的计算时间。

```python
import time

a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")

# 预热:第一次调用包含 kernel 编译/缓存建立等一次性开销,不能计入测量
for _ in range(3):
    _ = a @ b
torch.cuda.synchronize()

# 错误测法:没有等 GPU 算完
t0 = time.time()
c = a @ b
t_wrong = (time.time() - t0) * 1000

# 正确测法:synchronize() 阻塞 CPU 直到 GPU 队列里的活全部干完
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
t_right = (time.time() - t0) * 1000

print(f"不加 synchronize: {t_wrong:8.3f} ms   <- 只是'提交命令'的时间")
print(f"加 synchronize  : {t_right:8.3f} ms   <- GPU 真实计算时间")
print(f"两者相差 {t_right / max(t_wrong, 1e-9):.0f} 倍")
```

### 例 3:规范计时 —— `torch.cuda.Event`

`time.time()` + `synchronize` 能用,但会把 CPU 端的开销也算进去。更精确的做法是用 **CUDA Event**:往 GPU 命令队列里插两个"时间戳标记",由 GPU 自己记录,测出来的就是纯 GPU 耗时。下面这个 `gpu_time_ms` 函数本系列后面会反复使用(第 08 篇会升级为 `triton.testing.do_bench`)。

```python
def gpu_time_ms(fn, warmup=5, repeat=20):
    """用 CUDA Event 测量 fn() 在 GPU 上的中位数耗时(毫秒)。"""
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
        times.append(start.elapsed_time(end))  # 单位:毫秒
    times.sort()
    return times[len(times) // 2]  # 中位数比平均值更抗干扰


ms = gpu_time_ms(lambda: a @ b)
flops = 2 * 4096**3  # 矩阵乘的浮点运算次数:2*M*N*K
print(f"4096x4096 fp32 矩阵乘: {ms:.3f} ms, 实测算力 {flops / ms / 1e9:.1f} GFLOPS")
```

### 例 4:数据搬运不是免费的 —— PCIe 传输 vs GPU 计算

```python
n = 4096
x = torch.randn(n, n)  # 在 CPU 上

t_h2d = gpu_time_ms(lambda: x.to("cuda"))                      # Host -> Device 拷贝
x_gpu2 = x.to("cuda")
t_compute = gpu_time_ms(lambda: x_gpu2 @ x_gpu2)               # 纯 GPU 计算
t_d2h = gpu_time_ms(lambda: x_gpu2.cpu())                      # Device -> Host 拷贝

size_mb = x.numel() * x.element_size() / 1024**2
print(f"数据量: {size_mb:.0f} MB")
print(f"CPU->GPU 拷贝: {t_h2d:7.3f} ms (带宽 {size_mb / t_h2d:.1f} GB/s,受 PCIe 限制)")
print(f"GPU 矩阵乘   : {t_compute:7.3f} ms")
print(f"GPU->CPU 拷贝: {t_d2h:7.3f} ms")
print("\n结论:能留在 GPU 上的数据就别来回搬——推理框架(vLLM 等)的 KV Cache 常驻显存就是这个道理。")
```

---

## §3 知识连接

**与已有知识的联系:**

- 如果你用过 **NumPy**:`torch.Tensor` 的 API 与 `np.ndarray` 高度相似,核心差异就是多了 `device` 和自动求导;`tensor.numpy()` / `torch.from_numpy()` 可零拷贝互转(仅 CPU)。
- **异步队列模型**并不是 GPU 独有:它和"往消息队列里发任务、不等结果就返回"是同一种思想。CUDA Stream 就是 GPU 的任务队列。

**与后续篇章的联系:**

- 第 02 篇会解释:GPU 拿到 kernel 之后,内部的 SM、warp、显存层次是怎么干活的;
- 第 03 篇开始写 Triton:你会看到一个 kernel launch 在 Triton 里长什么样(`kernel[grid](...)`);
- 第 06 篇算子融合:本篇例 2 里"3 个小 kernel 浪费带宽"的问题在那里被正面解决;
- 第 08 篇 benchmark 方法论:`gpu_time_ms` 会升级为 `triton.testing.do_bench`。

**与真实框架的联系:**

- 本篇的 Event 计时写法,与 vLLM 仓库 `benchmarks/` 目录下的延迟测量脚本、PyTorch 官方 `torch.utils.benchmark` 模块的内部实现是同一套机制;
- `torch.cuda.synchronize` 在几乎所有推理框架的 benchmark 代码里都能找到——看到没加 sync 的 GPU 计时报告,你应该本能地怀疑它。

---

## §4 闭环对比实验

### 实验 A:CPU vs GPU —— 加速比与"交叉点"

GPU 不是永远更快:小规模计算时,kernel 启动和调度开销会吃掉所有收益。我们扫一遍矩阵规模,找出 CPU 和 GPU 的交叉点。

```python
import time

import matplotlib.pyplot as plt

sizes = [64, 128, 256, 512, 1024, 2048, 4096]
cpu_ms, gpu_ms = [], []

for n in sizes:
    xc = torch.randn(n, n)
    xg = xc.to("cuda")

    # CPU 计时(同步执行,直接用 time 即可)
    for _ in range(3):
        _ = xc @ xc
    t0 = time.time()
    reps = 10 if n <= 1024 else 3
    for _ in range(reps):
        _ = xc @ xc
    cpu_ms.append((time.time() - t0) / reps * 1000)

    # GPU 计时(CUDA Event)
    gpu_ms.append(gpu_time_ms(lambda: xg @ xg))

print(f"{'规模':>8} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'加速比':>8}")
for n, tc, tg in zip(sizes, cpu_ms, gpu_ms):
    print(f"{n:>8} | {tc:>10.3f} | {tg:>10.3f} | {tc / tg:>7.1f}x")

plt.figure(figsize=(8, 4))
plt.loglog(sizes, cpu_ms, "o-", label="CPU")
plt.loglog(sizes, gpu_ms, "s-", label="GPU (T4)")
plt.xlabel("matrix size N (NxN @ NxN)")
plt.ylabel("time (ms)")
plt.title("CPU vs GPU matmul")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()
```

### 实验 B:kernel 启动开销 —— 同样的计算量,切成 N 份提交

把"对 3200 万个元素做一次加法"这个固定计算量,分别切成 1 / 8 / 64 / 512 / 4096 个小 kernel 提交,总计算量完全相同,只有启动次数不同。

```python
total = 32 * 1024 * 1024  # 固定总元素数
chunks_list = [1, 8, 64, 512, 4096]
results = []

big = torch.randn(total, device="cuda")

for n_chunks in chunks_list:
    chunk = total // n_chunks
    pieces = [torch.randn(chunk, device="cuda") for _ in range(n_chunks)]

    def run_chunks(pieces=pieces):
        for p in pieces:
            p.add_(1.0)  # 每个 piece 一次独立 kernel launch

    ms = gpu_time_ms(run_chunks, warmup=3, repeat=10)
    results.append(ms)
    per_launch_us = ms / n_chunks * 1000
    print(f"切成 {n_chunks:>5} 个 kernel: 总耗时 {ms:8.3f} ms, 平均每次启动 {per_launch_us:8.2f} us")

plt.figure(figsize=(8, 4))
plt.semilogx(chunks_list, results, "o-")
plt.xlabel("number of kernel launches (same total work)")
plt.ylabel("total time (ms)")
plt.title("Kernel launch overhead: same FLOPs, more launches = slower")
plt.grid(True, alpha=0.3)
plt.show()
```

### 实验结果解读

- **实验 A**:小矩阵(64~256)时 GPU 相比 CPU 优势很小甚至更慢——时间被启动/调度开销主导;矩阵够大后 GPU 加速比拉开到几十倍。**GPU 是吞吐机器,不是低延迟机器**。
- **实验 B**:总计算量一模一样,切成 4096 个小 kernel 比 1 个大 kernel 慢一个数量级以上。每次启动约几微秒~几十微秒的固定开销,在小任务上是致命的。
- 这两个实验合起来就是 Triton 的"立项理由":**把多个小操作融合成一个大 kernel,减少启动次数和显存往返**——第 06 篇我们将亲手做到这一点。

---

## §5 练习 + 面试考点

### 动手练习

1. 修改实验 B:把 `add_(1.0)` 换成 `mul_(2.0).add_(1.0)`(两次 kernel)与等价的单次 `x.mul(2).add(1)` 链式调用,观察 kernel 数翻倍对耗时的影响。提示:用 `torch.cuda.synchronize()` 包好计时边界。
2. 用 `torch.profiler.profile`(`with profile(activities=[ProfilerActivity.CUDA]) as prof:`)跑一次 `a @ b`,在输出表里找出真正执行的 cuBLAS kernel 名字。

### 面试高频考点

- **Q:为什么 GPU 程序计时必须 synchronize?**
  A:CUDA 执行是异步的,CPU 提交 kernel 后立即返回;不同步测到的只是提交开销。规范做法是 CUDA Event 或带 sync 的计时,且必须先预热排除首次编译/缓存开销。
- **Q:一个 GPU 上的操作,时间可能花在哪几个地方?**
  A:① CPU 端调度与 kernel 启动开销;② GPU 显存读写(memory-bound);③ GPU 计算(compute-bound);④ Host↔Device 的 PCIe 传输。优化前先定位主要矛盾——第 02 篇 roofline 模型给出系统化判断方法。
- **Q:为什么深度学习框架要把很多小算子融合(fusion)?**
  A:每个算子独立启动 kernel,有固定启动开销,且中间结果要写回显存再读出来。融合后一次启动、中间结果留在寄存器/片上缓存,省启动开销更省显存带宽(实验 B 是直接证据)。
