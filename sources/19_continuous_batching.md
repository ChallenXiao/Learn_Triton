# 19 · Continuous Batching:推理吞吐的调度革命

> Learn Triton 系列 · 阶段 3(推理优化)第 5 篇
> 前置:第 15 篇(batch 是 decode 吞吐的杠杆;prefill/decode 两阶段)
> 运行环境:任意(本篇是调度层,纯 Python 模拟器,无需 GPU kernel;Colab CPU 亦可运行)

前几篇都在 kernel 层使劲;本篇上到**调度层**——它决定 kernel 收到的 batch 长什么样。第 15 篇证明了"batch 大 = 吞吐高",但真实流量是**长短不一、随时到达**的请求流,静态攒批根本吃不到那条曲线。**Continuous batching(迭代级调度,Orca OSDI 2022 提出,vLLM/SGLang/TensorRT-LLM 全部采用)** 是解法。本篇写一个迷你调度模拟器,把 static、continuous、chunked-prefill 三种策略放在同一请求流下对打,用吞吐/延迟/GPU 空泡三个维度闭环。

> 诚实声明:本篇实验是**事件级模拟**(成本模型来自第 15 篇的实测规律),不是真 GPU 执行——调度策略的对比本质是排队论问题,模拟是该领域的标准研究方法(Orca/vLLM 论文亦如此)。模型参数可换成你在第 15 篇测出的真实数字。

## 环境准备

```python
import heapq
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

random.seed(42)
print("纯 Python 模拟器,就绪")
```

---

## §1 是什么 & 能力边界

### static batching 的两种浪费

```text
static(请求级批处理):攒满 B 个请求 -> 一起 prefill -> 一起 decode 直到全部结束

浪费 1(尾部空泡):batch 里有人生成 10 token、有人生成 1000 token,
   短请求早早结束,但它的"坑位"要等最长那个请求跑完才释放
   -> decode 有效 batch 从 B 衰减到 1,吞吐曲线(第 15 篇)从右端滑回左端

浪费 2(队头阻塞):新请求来了,哪怕 GPU 里只剩 1 个活跃请求,也要等整批结束才能进
   -> TTFT 爆炸
```

### continuous batching:迭代级调度

把调度粒度从"一批请求"细化到"**一次前向迭代**":

```text
每个 decode step 之间都是调度点:
  - 谁生成完了 EOS -> 立刻离场,释放 KV(配合第 18 篇 paged,块立刻回池)
  - 等待队列有新请求且显存够 -> 立刻插入(先跑它的 prefill)
  - batch 成员每一步都在变 -> decode batch 始终保持饱满
```

前提条件正是前几篇打好的地基:**变长 batch 的 attention kernel**(第 17/18 篇的 seqlen 数组 + paged KV)——没有这些 kernel,迭代级调度无从谈起。系统问题和 kernel 问题在这里咬合。

### chunked prefill:消除 prefill 的"插队卡顿"

continuous batching 还剩一个毛刺:新请求插入时要跑完整 prefill(几百 ms),这一步里所有 decode 中的请求被迫等待 → TPOT 周期性尖刺。**chunked prefill**(Sarathi/vLLM)把长 prefill 切成小段,每个迭代"一小段 prefill + 全部 decode"混合执行——prefill 摊开,decode 平滑,代价是 TTFT 略增。

### 能做什么 / 不能做什么

能做:

- 吞吐 2~10 倍于 static(取决于输出长度方差——方差越大,static 浪费越多);
- TTFT 大幅下降(随到随进);与 paged KV、Flash-Decoding 正交叠加;
- chunked prefill 进一步把 P99 TPOT 的尖刺磨平。

不能做:

- 不突破物理上限:第 15 篇的"带宽墙"吞吐上限仍在,调度只是**让真实负载贴近那条上限曲线**;
- 显存仍是硬约束:batch 能开多大由 KV 池决定(所以与第 18 篇是乘法关系);
- 每步调度有 CPU 开销:调度器本身可能成为瓶颈(vLLM v1 重构的主因之一);
- batch 成员每步在变 → 无法用 CUDA Graph 一录到底,需要按 batch 段位预录(框架的工程细节)。

---

## §2 递进式例子

### 例 1:请求流 + 成本模型 —— 模拟器的地基

```python
@dataclass
class Request:
    rid: int
    arrival: float          # 到达时刻 (ms)
    prompt_len: int
    output_len: int         # 该请求将生成多少 token(模拟器上帝视角)
    # 运行时状态
    generated: int = 0
    prefill_done: int = 0   # chunked prefill 用:已 prefill 的 token 数
    t_first_token: float = field(default=-1.0)
    t_finish: float = field(default=-1.0)


def make_workload(n=200, qps=8.0):
    """泊松到达 + 长尾长度分布(贴近真实流量)。"""
    reqs, t = [], 0.0
    for i in range(n):
        t += random.expovariate(qps / 1000.0)            # 到达间隔 (ms)
        reqs.append(Request(
            rid=i, arrival=t,
            prompt_len=min(int(random.expovariate(1 / 400)) + 16, 3000),
            output_len=min(int(random.expovariate(1 / 150)) + 4, 1024)))
    return reqs


# 成本模型(参数取自第 15 篇 T4 + 0.5B 的实测量级,可替换)
PREFILL_MS_PER_TOKEN = 0.30      # prefill 摊到每 token(compute-bound, 近似线性)
DECODE_BASE_MS = 3.0             # decode 一步的固定成本(读一遍权重, 带宽墙)
DECODE_MS_PER_SEQ = 0.08         # batch 内每多一个序列的边际成本(读各自 KV)
MAX_BATCH = 64                   # 显存允许的最大并发(KV 池容量,第 18 篇)

def step_cost(prefill_tokens, decode_seqs):
    """一次前向迭代的耗时:prefill token 数 + decode 序列数。"""
    return (PREFILL_MS_PER_TOKEN * prefill_tokens
            + (DECODE_BASE_MS + DECODE_MS_PER_SEQ * decode_seqs if decode_seqs or prefill_tokens == 0 else 0)
            + (DECODE_BASE_MS if decode_seqs == 0 and prefill_tokens > 0 else 0))

print("示例:纯 decode batch=32 一步 =", step_cost(0, 32), "ms;插入一个 512 prefill 的迭代 =",
      step_cost(512, 32), "ms  <- 这就是 prefill 卡顿")
```

### 例 2:static batching 调度器

```python
def run_static(reqs, batch_size=16):
    """攒满一批(或等待超时)-> 整批 prefill -> 整批 decode 到全员结束。"""
    queue = sorted(reqs, key=lambda r: r.arrival)
    clock, i, done, busy = 0.0, 0, [], []   # busy: (时段起, 时段止, 有效batch, 容量batch)
    while i < len(queue):
        batch = []
        while i < len(queue) and len(batch) < batch_size:
            if queue[i].arrival <= clock or not batch:
                clock = max(clock, queue[i].arrival)
                batch.append(queue[i]); i += 1
            else:
                break
        clock += step_cost(sum(r.prompt_len for r in batch), 0)          # 整批 prefill
        for r in batch:
            r.t_first_token = clock
        max_out = max(r.output_len for r in batch)
        for step in range(max_out):                                       # 整批 decode
            alive = sum(1 for r in batch if r.output_len > step)
            t0 = clock
            clock += step_cost(0, len(batch))     # 占着整批的坑(尾部空泡所在!)
            busy.append((t0, clock, alive, len(batch)))
        for r in batch:
            r.t_finish = r.t_first_token + sum(
                step_cost(0, len(batch)) for _ in range(r.output_len))
            done.append(r)
    return done, clock, busy
```

### 例 3:continuous batching(+ 可选 chunked prefill)调度器

```python
def run_continuous(reqs, chunk=None):
    """迭代级调度。chunk=None: 整段 prefill 插入;chunk=N: 每迭代最多 prefill N token。"""
    queue = sorted(reqs, key=lambda r: r.arrival)
    waiting, running, done, busy = [], [], [], []
    clock, i = 0.0, 0
    while len(done) < len(reqs):
        while i < len(queue) and queue[i].arrival <= clock:               # 收新请求
            waiting.append(queue[i]); i += 1
        # 接纳:显存(MAX_BATCH)允许就进
        while waiting and len(running) < MAX_BATCH:
            running.append(waiting.pop(0))
        if not running:
            clock = queue[i].arrival if i < len(queue) else clock + 1.0
            continue

        # 本迭代的工作内容
        prefill_tokens = 0
        for r in running:
            if r.prefill_done < r.prompt_len:
                take = (r.prompt_len - r.prefill_done) if chunk is None else min(chunk - prefill_tokens, r.prompt_len - r.prefill_done)
                if take > 0:
                    r.prefill_done += take
                    prefill_tokens += take
                if chunk is not None and prefill_tokens >= chunk:
                    break
        decode_seqs = [r for r in running if r.prefill_done >= r.prompt_len]

        t0 = clock
        clock += step_cost(prefill_tokens, len(decode_seqs))
        busy.append((t0, clock, len(decode_seqs) + (1 if prefill_tokens else 0), MAX_BATCH))

        for r in decode_seqs:                                              # decode 推进一步
            if r.generated == 0:
                r.t_first_token = clock
            r.generated += 1
        finished = [r for r in decode_seqs if r.generated >= r.output_len]
        for r in finished:                                                 # ★ 立刻离场释放坑位
            r.t_finish = clock
            running.remove(r); done.append(r)
    return done, clock, busy
```

---

## §3 知识连接

**与前面篇章:**

- 成本模型直接来自第 15 篇:`DECODE_BASE_MS` 是"每步读一遍权重"的带宽墙,`DECODE_MS_PER_SEQ` 是 KV 读取的边际成本——模拟器的物理合理性建立在那篇的实测之上;
- "请求立刻离场释放坑位"在真实系统里 = 第 18 篇 paged KV 的块即时回池——**没有分页,continuous batching 释放的显存是碎的,接不进新请求**,两者是共生关系;
- 变长 batch 每步都在变,kernel 必须接受 seqlen 数组而非规整矩阵——第 17/18 篇 kernel 早已这样写。

**与真实框架:**

- Orca(OSDI'22)提出 iteration-level scheduling 与 selective batching;vLLM 将其与 PagedAttention 结合发扬光大(`vllm/core/scheduler.py`);
- chunked prefill 源自 Sarathi(-Serve)论文,vLLM 以 `enable_chunked_prefill`(现已默认)、SGLang/TensorRT-LLM 均内置;调度目标的术语:**吞吐(tokens/s)、TTFT、TPOT/ITL、P99**——本篇实验全部产出;
- SGLang 的调度在此之上又加了 cache-aware 维度(优先调度能命中 radix cache 的请求,第 20 篇)。

---

## §4 闭环对比实验:三种调度策略 × 同一请求流

200 个泊松到达的长尾请求,对打三策略:static(B=16)、continuous、continuous+chunked(256)。产出四个指标 + GPU 利用率时间线。

```python
def metrics(done, total_ms, name):
    tput = sum(r.output_len for r in done) / (total_ms / 1000)
    ttft = sorted(r.t_first_token - r.arrival for r in done)
    e2e = sorted(r.t_finish - r.arrival for r in done)
    p = lambda arr, q: arr[int(len(arr) * q)]
    print(f"{name:28s} 吞吐 {tput:7.1f} tok/s | TTFT p50 {p(ttft,0.5):7.0f} p99 {p(ttft,0.99):8.0f} ms"
          f" | 端到端 p50 {p(e2e,0.5):7.0f} ms")
    return tput, p(ttft, 0.5), p(e2e, 0.5)


import copy
workload = make_workload(n=200, qps=8.0)

results, timelines = {}, {}
for name, runner in [
    ("static (B=16)",            lambda w: run_static(w, 16)),
    ("continuous",               lambda w: run_continuous(w, chunk=None)),
    ("continuous+chunked(256)",  lambda w: run_continuous(w, chunk=256)),
]:
    w = copy.deepcopy(workload)
    done, total, busy = runner(w)
    results[name] = metrics(done, total, name)
    timelines[name] = busy

# ---- GPU 空泡可视化:时间线上的"有效 batch / 容量"利用率 ----
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for ax, (name, busy) in zip(axes, timelines.items()):
    xs = [(b[0] + b[1]) / 2 for b in busy]
    util = [b[2] / b[3] for b in busy]
    ax.fill_between(xs, util, step="mid", alpha=0.6)
    ax.set_ylabel("batch util"); ax.set_ylim(0, 1.05); ax.set_title(name, fontsize=10)
axes[-1].set_xlabel("time (ms)")
plt.suptitle("GPU batch utilization timeline: bubbles = wasted capacity")
plt.tight_layout(); plt.show()

# ---- 汇总条形图 ----
names = list(results)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5))
a1.barh(names[::-1], [results[n][0] for n in names[::-1]]); a1.set_xlabel("throughput (tok/s)")
a2.barh(names[::-1], [results[n][1] for n in names[::-1]], color="tab:orange"); a2.set_xlabel("TTFT p50 (ms)")
plt.tight_layout(); plt.show()
```

### 实验结果解读

- **static 的时间线**布满"楼梯下行"形空泡:每批后期有效 batch 从 16 衰减到 1,平均利用率常常不到一半;吞吐与 TTFT 双输(队头阻塞让 p99 TTFT 高一个量级);
- **continuous** 时间线接近满格:坑位即释即补,decode batch 常驻高位——同一硬件、同一请求流,**吞吐数倍于 static**,且 TTFT 大降。这就是 vLLM 发布时"24x throughput"故事里调度侧的那一半(另一半是第 18 篇的显存利用率);
- **chunked prefill** 吞吐与 continuous 相当,但把"插队 prefill"造成的长迭代磨平(时间线上更平滑),换来更稳的 TPOT——典型的 P99 工程:平均值不变,尾延迟改善,代价是 TTFT 略增;
- 三个策略读的是**同一物理上限**(成本模型没变),差距全部来自"浪费多少"——调度优化的本质是逼近上限,而 kernel 优化(前几篇)是抬高上限。两条腿,缺一不可。

---

## §5 练习 + 面试考点

### 动手练习

1. 给模拟器加上**显存约束的真实版**:每个 running 请求占 `prompt_len+generated` 的 KV 配额,总量封顶(而非简单 MAX_BATCH 个数),复现"长上下文挤占并发"现象;再叠加第 18 篇的 paged(按块取整)对比利用率。
2. 实现 **SJF(短作业优先)**变体:waiting 队列按 `prompt_len` 排序接纳,观察 TTFT p50 改善与长请求饥饿(p99 恶化)——调度公平性的经典权衡。

### 面试高频考点

- **Q:continuous batching 为什么能大幅提升吞吐?**
  A:静态批处理有尾部空泡(短请求陪跑等长请求)与队头阻塞(新请求等整批)。迭代级调度在每个 decode step 边界换人:完成即走、到达即进,使 decode batch 始终饱满。由第 15 篇"吞吐≈batch 的近线性函数",利用率提升直接折算为吞吐;输出长度方差越大,收益越大。
- **Q:chunked prefill 解决什么问题?代价是什么?**
  A:新请求的整段 prefill 会让正在 decode 的请求停顿(TPOT 尖刺)。把 prefill 切片,每迭代混合"一片 prefill + 全部 decode",平滑尾延迟、并让 compute-bound 的 prefill 片与 memory-bound 的 decode 互补提升硬件利用率;代价是该请求 TTFT 增加、实现复杂度上升。
- **Q:continuous batching 对 kernel 提出了什么要求?**
  A:必须支持"参差 batch":每序列不同长度(seqlen 数组)、KV 非连续(paged block table)、batch 成员每步变化(难以静态 CUDA Graph)。所以它和 PagedAttention/变长 FlashAttention 是一套组合拳,调度与 kernel 协同设计。
- **Q:推理服务的核心指标有哪些?分别被什么决定?**
  A:TTFT(prefill 速度 + 排队,受调度和 chunked 策略影响)、TPOT/ITL(decode 带宽墙 + batch 干扰)、吞吐(batch 饱满度 × 单步效率)、P99(尾部行为:长 prefill 插队、抢占、重调度)。容量规划在 SLA 约束下最大化吞吐。
