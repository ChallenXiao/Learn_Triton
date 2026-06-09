# 15 · 大模型推理全景:prefill / decode、KV Cache 与性能指标

> Learn Triton 系列 · 阶段 3(推理优化)第 1 篇
> 前置:第 02 篇(roofline、GEMV 是 memory-bound)、第 14 篇(attention)
> 运行环境:Google Colab T4 GPU(本篇要下载一个 0.5B 模型,约 1GB)

从本篇起视角拉高:不再只看单个 kernel,而是看**一次完整的 LLM 推理到底发生了什么、时间花在哪、显存吃在哪**。本篇用真实模型(Qwen2.5-0.5B)实测建立推理的"解剖图谱"——之后 7 篇(算子、FlashDecoding、PagedAttention、Continuous Batching、RadixAttention、量化、框架)全部是对这张图谱上某个痛点的手术。

## 环境准备

```python
import time

import torch

assert torch.cuda.is_available(), "请在 Colab 选择 GPU 运行时"

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate"], check=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()
cfg = model.config
n_params = sum(p.numel() for p in model.parameters())
print(f"模型: {MODEL_ID}")
print(f"参数量 {n_params/1e9:.2f}B | layers={cfg.num_hidden_layers} | heads={cfg.num_attention_heads} "
      f"| kv_heads={cfg.num_key_value_heads} | hidden={cfg.hidden_size} | head_dim={cfg.hidden_size//cfg.num_attention_heads}")
```

---

## §1 是什么 & 能力边界

### 一次生成 = 一次 prefill + N 次 decode

```text
输入 prompt(比如 1000 个 token)
   │
   ▼
[prefill] 一次前向吃下全部 1000 token            ← 大矩阵乘,compute-bound
   │        顺便把每层的 K/V 存进 KV Cache
   ▼
[decode] 每步只算 1 个新 token,重复 N 次         ← 全是 GEMV,memory-bound
   │        每步:读全部权重 + 读全部 KV cache
   ▼
输出 N 个 token
```

两个阶段的性格完全相反(第 02 篇语言):

| | prefill | decode |
|---|---------|--------|
| 一次处理 | 整个 prompt(数百~数千 token) | 1 个 token |
| 矩阵形状 | `[batch×seq, hidden] @ 权重` → 大 GEMM | `[batch, hidden] @ 权重` → GEMV |
| 瓶颈 | **算力**(Tensor Core) | **显存带宽**(读权重 + 读 KV) |
| 对应指标 | TTFT(Time To First Token) | TPOT(Time Per Output Token) |

### KV Cache:用显存换计算

attention 里第 $t$ 个 token 要和**前面所有** token 的 K/V 做运算。不缓存的话,每生成一个 token 都要把前文全部重算一遍——生成 N 个 token 的总计算量是 $O(N^2)$ 次前向。缓存每层算好的 K/V 后,每步只算新 token 的那一份:$O(N)$。代价是显存:

$$\text{KV bytes} = 2 \times L \times H_{kv} \times d_{head} \times S \times B \times \text{sizeof(dtype)}$$

(2 = K 和 V,L 层数,$H_{kv}$ KV 头数,S 序列长,B batch。)长上下文、大 batch 下 KV cache 轻松超过模型本体——它正是 PagedAttention(第 18 篇)要管理的对象。

### decode 速度的物理上限(本篇最重要的公式)

decode 每步必须把**全部权重**从显存读一遍(batch 内共享一次读取):

$$\text{TPOT}_{\min} \approx \frac{\text{权重字节数} + \text{KV cache 字节数}}{\text{显存带宽}}$$

0.5B 模型 fp16 权重 ≈ 1GB,T4 带宽 320GB/s → **每 token 至少 ~3.1ms,单流上限 ~320 token/s**——与算力毫无关系,这就是"decode 是 memory-bound"的定量表述。本篇实验会把实测值和这个上限放在一起。

### 能做什么 / 不能做什么(KV Cache 这个机制本身)

能做:把生成复杂度从 $O(N^2)$ 前向降到 $O(N)$;支持流式输出;是后续一切推理优化(paged、prefix 复用、投机解码)的载体。

不能做:

- 不省 prefill(prefill 本来就一次算完);
- 显存按 token 线性增长,**并发数 × 上下文长度受显存硬约束**(第 18 篇解决碎片,第 21 篇靠量化压缩);
- 跨请求默认不共享(相同 system prompt 也各存一份——第 20 篇 RadixAttention 解决);
- batch 内不同长度序列的 cache 对齐浪费(第 19 篇 continuous batching 解决)。

---

## §2 递进式例子

### 例 1:KV Cache 的存在意义 —— 开关对比

```python
prompt = "请用一句话解释什么是注意力机制。"
inputs = tok(prompt, return_tensors="pt").to("cuda")

def generate_timed(use_cache, max_new=64):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             use_cache=use_cache, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return time.time() - t0, out

for _ in range(2):  # 预热
    generate_timed(True, 8)

t_cache, out = generate_timed(True)
t_nocache, _ = generate_timed(False)
print(f"生成 64 token:  use_cache=True  {t_cache:6.2f}s   use_cache=False {t_nocache:6.2f}s")
print(f"KV cache 带来 {t_nocache / t_cache:.1f}x 加速(不缓存时每个新 token 都重算全部前文)")
print("输出:", tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)[:80])
```

### 例 2:KV Cache 显存账本 —— 公式 vs 实测

```python
def kv_bytes_formula(seqlen, batch=1):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * seqlen * batch * 2  # fp16

long_prompt = "数据 " * 2000   # 约 2000+ token 的长输入
li = tok(long_prompt, return_tensors="pt").to("cuda")
S = li.input_ids.shape[1]

with torch.no_grad():
    out = model(**li, use_cache=True)
pkv = out.past_key_values
# 实测:遍历 cache 里所有张量加总字节数
measured = sum(t.numel() * t.element_size()
               for layer in pkv for t in layer)
print(f"序列长 {S} token:")
print(f"  公式估算 KV = {kv_bytes_formula(S)/1024**2:7.1f} MB")
print(f"  实测     KV = {measured/1024**2:7.1f} MB")
print(f"\n外推:batch=32、上下文 4096 时 KV = {kv_bytes_formula(4096, 32)/1024**3:.2f} GB"
      f"(模型本体才 {n_params*2/1024**3:.2f} GB!)")
print(f"注意本模型 kv_heads={cfg.num_key_value_heads} < heads={cfg.num_attention_heads},"
      f"这就是 GQA —— 专为压 KV 显存设计(第 17 篇)")
```

### 例 3:把 TTFT 和 TPOT 拆开测

```python
from transformers.cache_utils import DynamicCache

def measure_ttft_tpot(prompt_len=512, decode_steps=64):
    ids = torch.randint(100, 10000, (1, prompt_len), device="cuda")
    with torch.no_grad():
        # ---- prefill:一次前向吃下整个 prompt ----
        torch.cuda.synchronize(); t0 = time.time()
        out = model(ids, use_cache=True)
        torch.cuda.synchronize()
        ttft = (time.time() - t0) * 1000

        # ---- decode:逐 token ----
        cache = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(decode_steps):
            out = model(next_id, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        tpot = (time.time() - t0) * 1000 / decode_steps
    return ttft, tpot


for plen in [128, 512, 2048]:
    ttft, tpot = measure_ttft_tpot(plen)
    print(f"prompt={plen:>5} token:  TTFT={ttft:7.1f} ms ({ttft/plen:5.2f} ms/token)   TPOT={tpot:6.2f} ms")

weight_gb = n_params * 2 / 1e9
print(f"\nTPOT 物理下限 ≈ 权重 {weight_gb:.2f}GB / 320GB/s = {weight_gb/320*1000:.2f} ms")
print("-> prefill 摊到每 token 比 decode 便宜一个量级(并行 vs 串行);TPOT 受带宽墙压制,与 prompt 长度近乎无关(短上下文时)")
```

---

## §3 知识连接

**与前面篇章:**

- 第 02 篇例 4 算过 GEMV(M=1)的 AI≈1、利用率 <1%——decode 的每个 Linear 层正是它;TPOT 下限公式就是 roofline 带宽墙的整模型版本;
- 第 14 篇 FlashAttention 主战场在 prefill(M=N 的大 attention);decode 的 attention 是"1 个 query 对全部 KV",形态完全不同(第 17 篇 Flash-Decoding 处理);
- 例 1 的 use_cache 开关是"计算换存储"母题(第 06/13 篇)的反向应用:**存储换计算**。

**与真实框架:**

- vLLM/SGLang/TensorRT-LLM 的 benchmark 报告全部围绕本篇三指标:TTFT、TPOT(或 ITL)、吞吐(tokens/s);
- HuggingFace 的 `DynamicCache` 即本篇实测对象;vLLM 把它换成分页式(第 18 篇)、SGLang 换成 radix tree 管理(第 20 篇);
- 本篇的"账本式分析"(权重字节 / 带宽 / KV 增长)是推理容量规划(capacity planning)面试题的标准解法。

---

## §4 闭环对比实验:batch size 是 decode 吞吐的免费午餐

decode 每步读一遍权重,**batch 内所有请求共享这次读取**:batch 翻倍 → 每 token 的权重读取摊薄一半 → 吞吐近乎翻倍,直到撞上算力/KV 带宽。实测 batch 1→64 的 decode 吞吐曲线。

```python
import matplotlib.pyplot as plt

def decode_throughput(batch, steps=32, ctx=256):
    ids = torch.randint(100, 10000, (batch, ctx), device="cuda")
    with torch.no_grad():
        out = model(ids, use_cache=True)
        cache = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(steps):
            out = model(next_id, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_id = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
    dt = time.time() - t0
    return batch * steps / dt          # 总 tokens/s


batches = [1, 2, 4, 8, 16, 32, 64]
tput = []
for b in batches:
    tp = decode_throughput(b)
    tput.append(tp)
    print(f"batch={b:>3}: {tp:8.1f} tokens/s   (人均 {tp/b:6.1f} tok/s)")

plt.figure(figsize=(9, 4.5))
plt.plot(batches, tput, "o-", label="measured throughput")
plt.plot(batches, [tput[0] * b for b in batches], "--", alpha=0.5, label="perfect linear scaling")
plt.xscale("log", base=2); plt.yscale("log")
plt.xlabel("batch size"); plt.ylabel("decode throughput (tokens/s)")
plt.title(f"{MODEL_ID}: decode throughput vs batch (T4)")
plt.legend(); plt.grid(True, which="both", alpha=0.3)
plt.show()
```

### 实验结果解读

- 小 batch 区吞吐随 batch **近线性增长**——权重读取被摊薄,几乎白拿的吞吐。这是第 02 篇"GEMV → 攒大 M"结论的整模型实证,也是一切 batching 优化的物理基础;
- batch 增大后曲线弯头出现:KV cache 读取(随 batch 线性增长,无法摊薄)与算力开始接管瓶颈;每请求延迟(TPOT)也在缓慢上升——**吞吐与延迟的交换**从此开始;
- 但现实没这么美好:真实请求**长短不一、随时到达**,静态攒批会让短请求等长请求、新请求等整批结束。怎么把这条曲线在动态负载下吃到——正是第 19 篇 **continuous batching** 的全部内容;
- 本篇图谱总结:prefill 看算力(已被 FlashAttention 拿下),decode 看带宽(量化第 21 篇压字节、攒批第 19 篇摊读取),显存看 KV(分页第 18 篇、共享第 20 篇)。阶段 3 的地图画完了。

---

## §5 练习 + 面试考点

### 动手练习

1. 把例 3 的 decode 改为测"不同上下文长度"(ctx=128~4096)下的 TPOT,验证:短上下文时 TPOT 由权重主导几乎不变,长上下文后 KV 读取占比上升、TPOT 开始爬坡。对照例 2 的公式解释拐点位置。
2. 用 `torch.profiler` 抓一步 decode,统计 Linear(GEMV)与 attention 的时间占比,和"权重字节 vs KV 字节"的比值对照——验证 memory-bound 的时间分布可以从字节账直接预测。

### 面试高频考点

- **Q:prefill 和 decode 的本质区别?分别怎么优化?**
  A:prefill 并行处理整个 prompt,是大 GEMM,compute-bound,优化靠 FlashAttention、算子融合、更高算力精度(FP8);decode 每步一个 token,是 GEMV,memory-bound,优化靠攒批(continuous batching)、量化压权重/KV 字节、投机解码摊薄读取。两阶段在调度上还会互相干扰(chunked prefill 缓解,第 19 篇)。
- **Q:估算一下:7B fp16 模型在 A100(2TB/s)上单流 decode 上限?**
  A:权重 14GB / 2TB/s ≈ 7ms/token ≈ 140 token/s(忽略 KV)。这类心算题考察"decode = 每 token 读一遍权重"的直觉,公式:TPOT ≥ 权重字节/带宽。
- **Q:KV cache 多大?写出公式。**
  A:2 × 层数 × KV头数 × head_dim × 序列长 × batch × 字节数。重点:与 batch 和序列长**双线性**;GQA 把 KV 头数从 H 压到 H_kv(如 32→8)直接省 4 倍;MLA(DeepSeek)用低秩压缩省更多。
- **Q:为什么大 batch 能提高 decode 吞吐?上限在哪?**
  A:权重读取一次服务整批,字节/Token 摊薄;上限来自:① KV 读取随 batch 线性增长不可摊薄;② M 增大后逐渐 compute-bound;③ 显存装不下更多 KV;④ 延迟 SLA。容量规划就是在这四条约束里找运营点。
