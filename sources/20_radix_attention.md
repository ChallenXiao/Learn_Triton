# 20 · RadixAttention 与 prefix caching:KV 复用的艺术

> Learn Triton 系列 · 阶段 3(推理优化)第 6 篇
> 前置:第 15 篇(KV cache、TTFT)、第 18 篇(paged KV、引用计数与 COW)
> 运行环境:Google Colab T4 GPU(实验 B 需要第 15 篇的 0.5B 模型)

第 18 篇结尾留了一个钩子:相同前缀的请求可以共享物理块。把这件事**系统化、自动化**,就是 **prefix caching**;把它做到极致的数据结构,就是 SGLang 的 **RadixAttention**(radix tree 管理所有历史请求的 KV,自动发现任意可复用前缀)。在"系统提示词动辄上千 token、Agent 反复调用、多轮对话"的今天,这是 TTFT 优化的第一杠杆。本篇实现一棵 token 级 radix tree + 与 paged KV 的对接,并用**真实模型**实测前缀复用的 TTFT 收益。

## 环境准备

```python
import torch

assert torch.cuda.is_available(), "请在 Colab 选择 GPU 运行时"

print(f"PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)}")
```

---

## §1 是什么 & 能力边界

### 哪里来这么多重复前缀?

```text
场景 1 系统提示词:所有用户共享同一段 1000+ token 的 system prompt
场景 2 多轮对话:  第 N 轮的前缀 = 前 N-1 轮的完整历史(逐轮线性增长)
场景 3 few-shot:  同一组示例 + 不同问题
场景 4 并行采样:  同一 prompt 生成 n 个候选(best-of-n、MCTS、Agent 树搜索)
场景 5 RAG:      固定模板 + 检索片段 + 问题
```

这些前缀的 KV **每次都重新 prefill 一遍**是纯浪费:prefill 是 compute-bound(第 15 篇),前缀几千 token 意味着几百 ms 的 TTFT 与可观的算力。

### prefix caching 的两个层次

**层次一(手动)**:应用层自己留住 `past_key_values`,下次拼接。HF transformers 能做,但只能"一个 cache 对一个续写",跨请求、跨用户无法自动匹配。

**层次二(自动,RadixAttention)**:推理引擎维护一棵 **radix tree(基数树/压缩前缀树)**:

```text
- 边:token 序列片段;节点:对应"该前缀的 KV 已在显存"(指向 paged 块,第 18 篇)
- 新请求到达:沿树匹配最长公共前缀 -> 命中部分零成本复用,只 prefill 剩余后缀
- 节点带引用计数:活跃请求引用的不可逐出;空闲节点按 LRU 逐出回收显存
- 生成结束:该请求的新 KV 也插进树,成为后人的"前缀"
```

radix tree(而非普通 trie)的意义:边上存**片段**而不是单 token,节点数 ∝ 分叉数而非 token 数,树本身极小。

### 能做什么 / 不能做什么

能做:

- 命中前缀的 prefill **完全免除**:TTFT 从"整个 prompt"降为"未命中后缀"(场景 1 常见 5~10x TTFT 提升);
- 显存去重:N 个并发请求共享同一前缀只存一份 KV(第 18 篇引用计数);
- 与调度协同:cache-aware scheduling(优先调度命中率高的请求,SGLang 的核心调度策略);
- 多轮对话场景吞吐大增(每轮只 prefill 新增部分)。

不能做:

- **只加速 prefill/TTFT,不加速 decode**(decode 的瓶颈是权重与自身 KV 读取,第 15 篇);
- 只能**精确 token 匹配**:prompt 里一个时间戳、一个用户名变了,其后全部失配——提示词工程要把"可变部分放后面"(这是真实的工程规范);
- KV 常驻显存才有得复用:显存紧张时被 LRU 逐出,命中率取决于"工作集 vs KV 池"(实验 C 模拟);跨机复用需要 KV 迁移(PD 分离、CacheGen 等进阶话题);
- 复用别人的 KV 有一致性前提:相同模型、相同精度;position 编码也必须一致(RoPE 是绝对位置起算的,前缀位置相同才可复用——这就是为什么只能"前缀"而不能"中段")。

---

## §2 递进式例子

### 例 1:token 级 radix tree —— 插入与最长前缀匹配

```python
class RadixNode:
    __slots__ = ["edges", "kv_handle", "ref", "last_used"]
    def __init__(self):
        self.edges = {}        # first_token -> (token_tuple, child)
        self.kv_handle = None  # 教学版:存"这段前缀的 KV 块编号列表"
        self.ref = 0
        self.last_used = 0.0


class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def match_prefix(self, tokens):
        """返回 (命中的 token 数, 沿途节点列表)。"""
        node, i, path = self.root, 0, []
        while i < len(tokens):
            e = node.edges.get(tokens[i])
            if e is None:
                break
            seg, child = e
            j = 0
            while j < len(seg) and i + j < len(tokens) and seg[j] == tokens[i + j]:
                j += 1
            if j < len(seg):                   # 边走了一半:部分命中,教学版按未命中处理(生产版会分裂边)
                break
            i += len(seg)
            node = child
            path.append(node)
        return i, path

    def insert(self, tokens):
        """把整条 token 序列插入树(必要时分裂边)。"""
        node, i = self.root, 0
        while i < len(tokens):
            e = node.edges.get(tokens[i])
            if e is None:                      # 全新分支
                child = RadixNode()
                node.edges[tokens[i]] = (tuple(tokens[i:]), child)
                return child
            seg, child = e
            j = 0
            while j < len(seg) and i + j < len(tokens) and seg[j] == tokens[i + j]:
                j += 1
            if j == len(seg):                  # 整条边匹配,继续往下
                node, i = child, i + len(seg)
            else:                              # ★ 分裂边:公共部分成为新中间节点
                mid = RadixNode()
                node.edges[tokens[i]] = (seg[:j], mid)
                mid.edges[seg[j]] = (seg[j:], child)
                node, i = mid, i + j
        return node


tree = RadixTree()
sys_prompt = list(range(1000, 1100))                     # 模拟 100 token 的系统提示词
tree.insert(sys_prompt + [1, 2, 3])                      # 请求 A
tree.insert(sys_prompt + [7, 8, 9, 10])                  # 请求 B(共享系统提示词)

hit, _ = tree.match_prefix(sys_prompt + [7, 8, 99])      # 新请求 C
print(f"请求 C(共 103 token)命中前缀 {hit} token -> 只需 prefill {103 - hit} 个")
hit2, _ = tree.match_prefix([5, 5, 5])
print(f"无关请求命中 {hit2} token(树根分叉,互不干扰)")
print("radix 特性:A、B 共享的 100 token 在树中只占 1 条边,分叉只在 [1..] / [7..] 处发生")
```

### 例 2:radix tree × paged KV —— 共享物理块与引用计数

把第 18 篇的 PagedKV 池接到树上:节点的 `kv_handle` 存物理块号;命中即直接借用块号拼 block table,**一字节 KV 都不重算/重存**。

```python
BLOCK_TOKENS = 16

class CachedEngine:
    """教学版:演示 '命中复用块,未命中分配新块' 的核算逻辑。"""
    def __init__(self, num_blocks=4096):
        self.tree = RadixTree()
        self.free_blocks = num_blocks
        self.allocated = 0

    def admit(self, tokens):
        hit, path = self.tree.match_prefix(tokens)
        hit_blocks = hit // BLOCK_TOKENS                    # 整块命中才可复用(块是共享粒度)
        need_tokens = len(tokens) - hit_blocks * BLOCK_TOKENS
        need_blocks = -(-need_tokens // BLOCK_TOKENS)
        self.free_blocks -= need_blocks
        self.allocated += need_blocks
        node = self.tree.insert(tokens)
        return hit_blocks * BLOCK_TOKENS, need_tokens


eng_cached, eng_plain = CachedEngine(), CachedEngine()
sys_p = list(range(2000, 2960))                              # 960-token 系统提示词
total_prefill_cached = total_prefill_plain = 0
for u in range(50):                                          # 50 个用户请求,各带不同问题
    query = [10_000 + u * 13 + k for k in range(40)]
    hit, todo = eng_cached.admit(sys_p + query)
    total_prefill_cached += todo
    total_prefill_plain += len(sys_p) + len(query)           # 无缓存:全量 prefill

print(f"50 个共享系统提示词的请求:")
print(f"  无 prefix cache: 累计 prefill {total_prefill_plain:>7,} token, 占用块 {50 * -(-1000 // BLOCK_TOKENS)}")
print(f"  RadixAttention : 累计 prefill {total_prefill_cached:>7,} token, 占用块 {eng_cached.allocated}")
print(f"  -> prefill 计算省 {(1 - total_prefill_cached / total_prefill_plain) * 100:.0f}%,"
      f" KV 显存省 {(1 - eng_cached.allocated / (50 * -(-1000 // BLOCK_TOKENS))) * 100:.0f}%")
```

### 例 3:复用的 KV 算出来的 attention 和重算完全一样吗?

一致性 sanity check:第 18 篇的 paged kernel,两个请求的 block table **指向同一批物理块**(共享前缀)+ 各自的私有块,输出应与各自独立存储完全一致。

```python
import torch.nn.functional as F
import triton
import triton.language as tl

# 复用第 18 篇的 paged decode attention kernel(此处重新定义,保持 notebook 自包含)
@triton.jit
def paged_decode_attn_kernel(q_ptr, k_pool_ptr, v_pool_ptr, table_ptr, seqlen_ptr,
                             o_ptr, max_blocks, sm_scale,
                             BLOCK_T: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_T)
    q = tl.load(q_ptr + pid * HEAD_DIM + offs_d).to(tl.float32)
    seqlen = tl.load(seqlen_ptr + pid)
    m = float("-inf"); l = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for b in range(0, tl.cdiv(seqlen, BLOCK_T)):
        phys = tl.load(table_ptr + pid * max_blocks + b)
        tok = b * BLOCK_T + offs_t
        mask_t = tok < seqlen
        k_blk = tl.load(k_pool_ptr + phys * BLOCK_T * HEAD_DIM + offs_t[:, None] * HEAD_DIM + offs_d[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        s = tl.where(mask_t, tl.sum(q[None, :] * k_blk, axis=1) * sm_scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new); p = tl.exp(s - m_new)
        l = alpha * l + tl.sum(p, axis=0)
        v_blk = tl.load(v_pool_ptr + phys * BLOCK_T * HEAD_DIM + offs_t[:, None] * HEAD_DIM + offs_d[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_blk, axis=0)
        m = m_new
    tl.store(o_ptr + pid * HEAD_DIM + offs_d, (acc / l).to(tl.float16))


D = 64
shared_blocks, priv_a, priv_b = 8, 4, 6        # 前 8 块共享(128 token 前缀)
pool = torch.randn(shared_blocks + priv_a + priv_b, BLOCK_TOKENS, D, device="cuda", dtype=torch.float16)
v_pool = torch.randn_like(pool)

max_blocks = shared_blocks + max(priv_a, priv_b)
table = torch.zeros(2, max_blocks, device="cuda", dtype=torch.int32)
table[0, :shared_blocks] = torch.arange(shared_blocks)                       # 请求 A:共享块 + 私有块
table[0, shared_blocks:shared_blocks + priv_a] = torch.arange(shared_blocks, shared_blocks + priv_a)
table[1, :shared_blocks] = torch.arange(shared_blocks)                       # 请求 B:同一批共享块!
table[1, shared_blocks:shared_blocks + priv_b] = torch.arange(shared_blocks + priv_a,
                                                              shared_blocks + priv_a + priv_b)
seqlens = torch.tensor([(shared_blocks + priv_a) * BLOCK_TOKENS,
                        (shared_blocks + priv_b) * BLOCK_TOKENS], device="cuda", dtype=torch.int32)

q = torch.randn(2, D, device="cuda", dtype=torch.float16)
o = torch.empty_like(q)
paged_decode_attn_kernel[(2,)](q, pool, v_pool, table, seqlens, o, max_blocks, D ** -0.5,
                               BLOCK_T=BLOCK_TOKENS, HEAD_DIM=D)

for i in range(2):
    L = seqlens[i].item()
    blocks = table[i, :L // BLOCK_TOKENS].long()
    k_cont, v_cont = pool[blocks].reshape(-1, D)[:L], v_pool[blocks].reshape(-1, D)[:L]
    ref = F.scaled_dot_product_attention(q[i][None, None, None, :].float(),
                                         k_cont[None, None].float(), v_cont[None, None].float())[0, 0, 0]
    torch.testing.assert_close(o[i].float(), ref, rtol=2e-2, atol=2e-2)
print("两个请求共享同一批物理 KV 块,attention 结果与独立存储完全一致 ✓")
print("-> 共享是纯指针操作,数值路径零差异;这就是 prefix caching '免费' 的原因")
```

---

## §3 知识连接

**与前面篇章:**

- 第 18 篇是地基:没有 paged(共享粒度=块)+ 引用计数 + COW,radix tree 只是一个查找结构;两篇合起来才是完整的 KV 管理子系统;
- 第 19 篇调度的进阶:SGLang 的 cache-aware scheduling = continuous batching 的接纳顺序按"radix 命中率"重排——三篇(18/19/20)构成推理系统内存-调度协同的全景;
- "只能精确前缀匹配"的根因是第 16 篇 RoPE:KV 编码了绝对位置,换了位置的 KV 不可复用。

**与真实框架:**

- SGLang:RadixAttention 是其论文(2024)的核心贡献,`python/sglang/srt/mem_cache/radix_cache.py` 即生产版 radix tree(token 级、可分裂边、LRU 逐出),配合其调度器的 cache-aware 策略;
- vLLM:`enable_prefix_caching` 采用"块哈希"方案(每个满块按内容+前缀哈希,等价于固定粒度的前缀树),与 radix 方案各有取舍(哈希实现简单,radix 匹配粒度细);
- HF transformers:层次一的手动复用(传 `past_key_values` 续写),本篇实验 B 直接用它实测收益;
- 进阶方向(面试加分):PD 分离(prefill/decode 分机)下的 KV 传输、LMCache/CacheGen 的 KV 压缩外存、多租户下的 cache 隔离。

---

## §4 闭环对比实验

### 实验 A(真实模型):前缀缓存对 TTFT 的影响

用第 15 篇的 Qwen2.5-0.5B:固定 1600-token 系统提示词 + 60-token 用户问题。对比"全量 prefill" vs "系统提示词 KV 已缓存,只 prefill 问题部分"。

```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda().eval()

sys_ids = torch.randint(100, 10000, (1, 1600), device="cuda")     # 模拟长系统提示词
qry_ids = torch.randint(100, 10000, (1, 60), device="cuda")

with torch.no_grad():
    # 预计算系统提示词的 KV(= radix tree 中已存在的节点)
    cached = model(sys_ids, use_cache=True).past_key_values

    def ttft_no_cache():
        out = model(torch.cat([sys_ids, qry_ids], dim=1), use_cache=True)
        return out

    def ttft_with_cache():
        # 命中 1600 token 前缀,只 prefill 60 token 的后缀
        import copy as _copy
        out = model(qry_ids, past_key_values=_copy.deepcopy(cached), use_cache=True)
        return out

    for fn in (ttft_no_cache, ttft_with_cache):   # 预热
        fn()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(5):
        ttft_no_cache()
    torch.cuda.synchronize()
    t_nc = (time.time() - t0) / 5 * 1000

    t0 = time.time()
    for _ in range(5):
        ttft_with_cache()
    torch.cuda.synchronize()
    t_wc = (time.time() - t0) / 5 * 1000

print(f"TTFT 全量 prefill(1660 tok): {t_nc:7.1f} ms")
print(f"TTFT 命中前缀(只算 60 tok) : {t_wc:7.1f} ms   -> {t_nc / t_wc:.1f}x 加速")
```

### 实验 B(模拟):多轮对话场景的命中率与节省

```python
import random
import matplotlib.pyplot as plt

random.seed(1)

def simulate_chat_workload(n_users=20, turns=8, sys_len=1000, msg_len=80):
    """多轮对话:每轮 prompt = 系统提示词 + 全部历史 + 新消息。统计 radix 命中。"""
    tree = RadixTree()
    saved, total = 0, 0
    hit_rates = []
    for u in range(n_users):
        history = [50_000 + u]                                # 每用户唯一标记
        sys_tokens = list(range(1, sys_len + 1))              # 共享系统提示词
        for t in range(turns):
            history += [60_000 + u * 100 + t * 7 + i for i in range(msg_len)]
            prompt = sys_tokens + history
            hit, _ = tree.match_prefix(prompt)
            tree.insert(prompt)
            saved += hit
            total += len(prompt)
            hit_rates.append(hit / len(prompt))
    return saved, total, hit_rates


saved, total, hit_rates = simulate_chat_workload()
print(f"多轮对话负载:总 prefill 需求 {total:,} token,radix 命中 {saved:,} token"
      f"({saved/total*100:.0f}%)—— 即 prefill 计算量直接砍掉 {saved/total*100:.0f}%")

plt.figure(figsize=(9, 3.5))
plt.plot(hit_rates, ".", alpha=0.5)
plt.xlabel("request #"); plt.ylabel("prefix hit rate")
plt.title("Radix prefix hit rate over a multi-turn chat workload")
plt.grid(True, alpha=0.3); plt.show()
```

### 实验结果解读

- **实验 A**:命中 96% 的前缀后 TTFT 提升通常 5~10 倍——prefill 时间 ≈ 正比于"要算的 token 数",缓存把 1660 变成 60。这就是为什么所有给"带长系统提示词的 API"做推理的团队都把 prefix caching 当一级特性;
- **实验 B**:多轮对话的命中率随轮次递增(每轮只有新消息未命中),稳态命中率 >80%——SGLang 在 Agent/对话类负载上的吞吐优势主要来自这里;
- 综合三篇(18/19/20):**paged 解决"存得下"、continuous batching 解决"喂得满"、radix/prefix 解决"不重算"**——这三件事正是 vLLM/SGLang 文档首页列的三大卖点,你现在每一件都亲手实现过教学版。

---

## §5 练习 + 面试考点

### 动手练习

1. 给例 1 的 RadixTree 补上 **LRU 逐出**:节点记录 `last_used` 与引用计数,实现 `evict(n_blocks)`(只逐出 ref==0 的叶子,自底向上)。用实验 B 的负载 + 有限块池,画"池大小 vs 命中率"曲线。
2. 把例 2 的"整块命中才复用"改进为 **COW**:命中到块中间时复制该块(第 18 篇),统计 COW 触发频率与额外拷贝量。

### 面试高频考点

- **Q:RadixAttention 是什么?和 PagedAttention 什么关系?**
  A:SGLang 提出的自动 KV 复用机制:radix tree 索引全部历史请求的 token 前缀,节点指向 paged KV 块;新请求自动匹配最长公共前缀,免去对应 prefill,LRU 管理逐出。Paged 提供共享的物理基础(块 + 引用计数),radix 提供"找到能共享什么"的索引——一个是内存管理,一个是缓存策略。
- **Q:prefix caching 能加速 decode 吗?**
  A:不能。它免除的是命中前缀的 prefill 计算(TTFT),decode 每步仍要读全部 KV 与权重。它还省显存(共享一份 KV),间接通过更大 batch 提升吞吐。
- **Q:为什么 KV 只能按"前缀"复用,不能复用中间片段?**
  A:① 位置编码:RoPE 把绝对位置写进 K,换位置即失效;② 因果性:第 i 个 token 的 KV 依赖前面全部内容,前缀不同则 KV 不同。所以缓存键是"从头开始的精确 token 序列"。工程推论:prompt 模板要把可变字段放尾部。
- **Q:vLLM 和 SGLang 的 prefix caching 实现差异?**
  A:vLLM 用块级内容哈希(每个满 16-token 块,按"自身+前缀"哈希查重),实现简单、粒度=块;SGLang 用 token 级 radix tree(可分裂边),匹配粒度细、天然支持树状分叉(Agent/并行采样),配合 cache-aware 调度。负载重前缀共享时 SGLang 思路收益更明显。
