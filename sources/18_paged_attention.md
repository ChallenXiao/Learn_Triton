# 18 · PagedAttention:给 KV Cache 装上虚拟内存

> Learn Triton 系列 · 阶段 3(推理优化)第 4 篇
> 前置:第 15 篇(KV cache 账本)、第 17 篇(decode attention kernel)、第 05 篇(gather)
> 运行环境:Google Colab T4 GPU

第 15 篇算过:KV cache 是推理显存的吞噬者。但比"大"更糟的是"**碎**":传统做法按最大长度预分配连续显存,真实请求长短不一,显存大量被浪费在"占着不用"上——vLLM 论文实测浪费 60%~80%。**PagedAttention**(vLLM 的成名作,SOSP 2023)把操作系统虚拟内存的思想搬进 KV cache:固定大小的物理块 + 逻辑到物理的 block table 映射。本篇实现它的核心:**带间接寻址的 paged decode attention kernel**,并用碎片模拟 + kernel 开销两个实验闭环。

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

### 问题:连续 KV cache 的三宗罪

```text
传统:每个请求预留 [max_len, H_kv, D] 的连续显存
罪 1 内部碎片:请求实际只生成 300 token,却占着 4096 的位置 → 92% 浪费
罪 2 外部碎片:长短请求交替分配/释放,显存被切成无法利用的零碎缝隙
罪 3 无法共享:两个请求有相同的 system prompt,KV 也得各存一份
```

浪费的直接后果:**能同时服务的请求数(并发度)被腰斩**——而第 15 篇实验证明,decode 吞吐 ≈ 正比于 batch size。显存利用率就是吞吐。

### 解法:分页(与操作系统教科书逐字对应)

```text
物理层:显存切成固定大小的 block(vLLM 默认 16 token/block)组成 block pool
逻辑层:每个请求只持有一张 block table:逻辑块号 -> 物理块号
        请求长 300 token -> 持有 ceil(300/16)=19 个块,浪费 < 1 个块
分配:  请求每长出 16 token,从 pool 取一个空闲块,append 到 table
共享:  相同前缀的请求,block table 指向同一批物理块(引用计数 + copy-on-write)
```

| OS 概念 | PagedAttention 对应 |
|---|---|
| 物理页帧 | KV block(16 token) |
| 页表 | block table |
| 缺页分配 | decode 时按需取块 |
| 共享页 + COW | 前缀共享 + 分叉时复制 |

### kernel 的代价:间接寻址

KV 不再连续,attention kernel 读第 $n$ 个 token 的 K 时要先查表:

```text
physical_block = block_table[n // BLOCK_SIZE]     ← 多一跳 load(第 05 篇的 gather)
k = K_pool[physical_block, n % BLOCK_SIZE, :]
```

好消息:块内 16 个 token 仍然连续(第 05 篇:行级 gather 不破坏合并访存),块表本身极小(常驻 L2)。实验会证明这层间接寻址的开销只有百分之几。

### 能做什么 / 不能做什么

能做:

- 显存利用率从 ~30% 提到 >90%,等量显存下并发数(≈吞吐)翻 2~4 倍;
- 动态增长,无需预知输出长度;支持前缀共享(第 20 篇 RadixAttention 在此之上建树);
- 抢占与换出:整块粒度可以把低优请求的 KV 换到 CPU 内存(vLLM 的 swapping)。

不能做:

- **不减少 KV 本身的字节数**(那是 GQA/量化/MLA 的事),只消灭浪费;
- 间接寻址有小开销,且 KV 写入(新 token 的 K/V 入块)也要走查表 scatter;
- block size 是权衡:太小则块表大、gather 碎;太大则内部碎片回升(最后一块平均浪费 size/2);
- 实现复杂度上升:分配器、引用计数、抢占逻辑——这正是"用 vLLM 而不是自己写"的理由,本篇教学版只做只读路径。

---

## §2 递进式例子

### 例 1:碎片有多严重 —— 显存利用率模拟

```python
import random

random.seed(0)

def simulate(memory_blocks=10_000, block_tokens=16, max_len=4096, n_req=10_000):
    """模拟两种分配策略下,固定显存能接纳的并发请求数与利用率。"""
    # 请求实际长度:长尾分布(多数短,少数长),贴近真实流量
    lengths = [min(int(random.expovariate(1 / 600)) + 32, max_len) for _ in range(n_req)]

    total_tokens_capacity = memory_blocks * block_tokens

    # 策略 A:连续预分配 max_len
    per_req_naive = max_len                       # 每个请求占 max_len 的位置
    fit_naive = total_tokens_capacity // per_req_naive
    used_naive = sum(lengths[:fit_naive])
    util_naive = used_naive / (fit_naive * per_req_naive)

    # 策略 B:paged,按需取块
    fit_paged, used_blocks = 0, 0
    for L in lengths:
        need = -(-L // block_tokens)              # ceil
        if used_blocks + need > memory_blocks:
            break
        used_blocks += need
        fit_paged += 1
    util_paged = sum(lengths[:fit_paged]) / (used_blocks * block_tokens)

    return fit_naive, util_naive, fit_paged, util_paged


fn, un, fp, up = simulate()
print(f"同样的显存预算(16 万 token 容量):")
print(f"  连续预分配: 并发 {fn:>4} 个请求, 显存利用率 {un*100:5.1f}%")
print(f"  Paged     : 并发 {fp:>4} 个请求, 显存利用率 {up*100:5.1f}%")
print(f"  -> 并发提升 {fp/fn:.1f}x。第 15 篇:并发≈吞吐,这就是 vLLM 吞吐碾压旧框架的主要来源")
```

### 例 2:搭一个 paged KV 存储 —— block table 的最小实现

```python
BLOCK_TOKENS = 16
D = 64

class PagedKV:
    """教学版 paged KV(单层单头):物理池 + 每请求 block table。"""
    def __init__(self, num_blocks):
        self.k_pool = torch.zeros(num_blocks, BLOCK_TOKENS, D, device="cuda", dtype=torch.float16)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free = list(range(num_blocks))
        self.tables = {}                          # req_id -> [物理块号,...]
        self.lens = {}

    def append(self, req, k_new, v_new):
        """给请求 req 追加若干 token 的 K/V(模拟 prefill 或 decode 写入)。"""
        table = self.tables.setdefault(req, [])
        pos = self.lens.get(req, 0)
        for i in range(k_new.shape[0]):
            blk_idx, off = (pos + i) // BLOCK_TOKENS, (pos + i) % BLOCK_TOKENS
            if blk_idx == len(table):
                table.append(self.free.pop())     # 缺"页"分配
            self.k_pool[table[blk_idx], off] = k_new[i]
            self.v_pool[table[blk_idx], off] = v_new[i]
        self.lens[req] = pos + k_new.shape[0]

    def gather(self, req):
        """把逻辑 KV 重新拼成连续张量(参考用,真实系统从不这样做)。"""
        t, L = self.tables[req], self.lens[req]
        k = self.k_pool[t].reshape(-1, D)[:L]
        v = self.v_pool[t].reshape(-1, D)[:L]
        return k, v


pkv = PagedKV(num_blocks=256)
k_ref = torch.randn(1000, D, device="cuda", dtype=torch.float16)
v_ref = torch.randn_like(k_ref)
pkv.append("req0", k_ref[:700], v_ref[:700])      # prefill 700 token
for i in range(700, 1000):                         # decode 逐 token 追加
    pkv.append("req0", k_ref[i:i+1], v_ref[i:i+1])

k_got, v_got = pkv.gather("req0")
torch.testing.assert_close(k_got, k_ref)
torch.testing.assert_close(v_got, v_ref)
print(f"paged 存储正确 ✓:1000 token 用了 {len(pkv.tables['req0'])} 个物理块"
      f"(浪费 {len(pkv.tables['req0'])*BLOCK_TOKENS - 1000} token 位 < 1 块)")
print(f"物理块号(乱序,不连续):{pkv.tables['req0'][:8]} ...")
```

### 例 3:paged decode attention kernel —— 间接寻址进 kernel

第 17 篇例 3 的 decode kernel,把"连续 KV"换成"查 block table 取页"。注意 BLOCK_N 取页大小,**一次循环恰好处理一页**。

```python
@triton.jit
def paged_decode_attn_kernel(
    q_ptr,                                  # [BH, D]
    k_pool_ptr, v_pool_ptr,                 # [num_blocks, BLOCK_T, D](所有请求共享的池)
    table_ptr,                              # [BH, max_blocks] 各请求的 block table
    seqlen_ptr,                             # [BH] 各请求当前 KV 长度
    o_ptr, max_blocks, sm_scale,
    BLOCK_T: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)                  # 一个 program 负责一个 (batch, head)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_T)

    q = tl.load(q_ptr + pid * HEAD_DIM + offs_d).to(tl.float32)
    seqlen = tl.load(seqlen_ptr + pid)
    n_blocks = tl.cdiv(seqlen, BLOCK_T)

    m = float("-inf"); l = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for b in range(0, n_blocks):
        phys = tl.load(table_ptr + pid * max_blocks + b)         # ★ 间接寻址:查页表
        tok = b * BLOCK_T + offs_t
        mask_t = tok < seqlen                                     # 最后一页可能不满
        k_blk = tl.load(k_pool_ptr + phys * BLOCK_T * HEAD_DIM +
                        offs_t[:, None] * HEAD_DIM + offs_d[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k_blk, axis=1) * sm_scale
        s = tl.where(mask_t, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        l = alpha * l + tl.sum(p, axis=0)
        v_blk = tl.load(v_pool_ptr + phys * BLOCK_T * HEAD_DIM +
                        offs_t[:, None] * HEAD_DIM + offs_d[None, :],
                        mask=mask_t[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_blk, axis=0)
        m = m_new

    tl.store(o_ptr + pid * HEAD_DIM + offs_d, (acc / l).to(tl.float16))


# ---- 构造一批请求:不同长度,块在池中乱序分布 ----
BH, max_blocks = 8, 512
pool_blocks = BH * max_blocks
k_pool = torch.randn(pool_blocks, BLOCK_TOKENS, D, device="cuda", dtype=torch.float16)
v_pool = torch.randn_like(k_pool)
perm = torch.randperm(pool_blocks, device="cuda")                # 模拟乱序分配
table = perm[:BH * max_blocks].reshape(BH, max_blocks).int()
seqlens = torch.tensor([700, 3000, 128, 5000, 4096, 17, 2048, 1234], device="cuda", dtype=torch.int32)

q = torch.randn(BH, D, device="cuda", dtype=torch.float16)
o = torch.empty_like(q)
paged_decode_attn_kernel[(BH,)](q, k_pool, v_pool, table, seqlens, o,
                                max_blocks, D ** -0.5,
                                BLOCK_T=BLOCK_TOKENS, HEAD_DIM=D, num_warps=4)

# 参考:把每个请求的 KV gather 成连续张量后做标准 attention
for i in range(BH):
    L = seqlens[i].item()
    blocks = table[i, :-(-L // BLOCK_TOKENS)].long()
    k_cont = k_pool[blocks].reshape(-1, D)[:L]
    v_cont = v_pool[blocks].reshape(-1, D)[:L]
    ref = F.scaled_dot_product_attention(q[i][None, None, None, :].float(),
                                         k_cont[None, None].float(), v_cont[None, None].float())[0, 0, 0]
    torch.testing.assert_close(o[i].float(), ref, rtol=2e-2, atol=2e-2)
print("paged decode attention 正确 ✓ —— 8 个变长请求,KV 块在池中完全乱序,逐请求对拍通过")
```

---

## §3 知识连接

**与前面篇章:**

- 例 3 = 第 17 篇 decode kernel + 第 05 篇例 3(gather/"指针的指针")——当时埋的伏笔在此兑现:**block table 就是工业级的 embedding lookup**;
- 变长 mask(`tok < seqlen`)是第 04 篇 mask 哲学处理"参差 batch"的标准姿势;
- 本篇解决"KV 占用浪费",与第 17 篇(decode 并行度)、第 15 篇(攒批)正交,三者在 vLLM 中叠加使用。

**与真实框架:**

- vLLM:本篇概念与 `vllm/core/block_manager.py`(分配器/引用计数)、`vllm/attention/ops/paged_attn.py` 及 CUDA kernel `csrc/attention/` 直接对应;`reshape_and_cache` kernel 负责写入路径(新 token KV scatter 进池);
- SGLang:同样基于 paged 池(其 token-level radix tree 把"页"细化到 token 级共享,第 20 篇);
- TensorRT-LLM:paged KV cache 同为标配;HuggingFace transformers 自 4.4x 起的 `StaticCache`/分页后端也吸收了该设计——**paged 已是行业默认,不再是 vLLM 专利**;
- OS 类比在面试中是加分叙事:页表/缺页/COW 三连讲清楚,等于同时回答了"为什么 block 固定大小"和"前缀共享怎么做"。

---

## §4 闭环对比实验:间接寻址的开销 vs 它换来的并发

**A. kernel 开销**:同一份 KV 数据,连续布局(第 17 篇 kernel)vs paged 布局(本篇 kernel),KV 长度扫描——量化"查页表"的真实代价。

```python
import matplotlib.pyplot as plt

@triton.jit
def contig_decode_attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, stride_bh, stride_n,
                              seqlen, sm_scale,
                              BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + pid * HEAD_DIM + offs_d).to(tl.float32)
    m = float("-inf"); l = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for start in range(0, seqlen, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen
        k_blk = tl.load(k_ptr + pid * stride_bh + offs_n[:, None] * stride_n + offs_d[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
        s = tl.where(mask_n, tl.sum(q[None, :] * k_blk, axis=1) * sm_scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=0))
        alpha = tl.exp(m - m_new); p = tl.exp(s - m_new)
        l = alpha * l + tl.sum(p, axis=0)
        v_blk = tl.load(v_ptr + pid * stride_bh + offs_n[:, None] * stride_n + offs_d[None, :],
                        mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v_blk, axis=0)
        m = m_new
    tl.store(o_ptr + pid * HEAD_DIM + offs_d, (acc / l).to(tl.float16))


kv_lens = [1024, 4096, 16384, 65536]
t_contig, t_paged = [], []
for S in kv_lens:
    nb = -(-S // BLOCK_TOKENS)
    kc = torch.randn(BH, S, D, device="cuda", dtype=torch.float16)
    vc = torch.randn_like(kc)
    kp = torch.randn(BH * nb, BLOCK_TOKENS, D, device="cuda", dtype=torch.float16)
    vp = torch.randn_like(kp)
    tb = torch.randperm(BH * nb, device="cuda").reshape(BH, nb).int()
    sl = torch.full((BH,), S, device="cuda", dtype=torch.int32)
    oc = torch.empty(BH, D, device="cuda", dtype=torch.float16)

    ms_c = triton.testing.do_bench(lambda: contig_decode_attn_kernel[(BH,)](
        q, kc, vc, oc, kc.stride(0), kc.stride(1), S, D ** -0.5,
        BLOCK_N=BLOCK_TOKENS, HEAD_DIM=D), return_mode="median")
    ms_p = triton.testing.do_bench(lambda: paged_decode_attn_kernel[(BH,)](
        q, kp, vp, tb, sl, oc, nb, D ** -0.5,
        BLOCK_T=BLOCK_TOKENS, HEAD_DIM=D), return_mode="median")
    t_contig.append(ms_c); t_paged.append(ms_p)
    print(f"KV={S:>6}: 连续 {ms_c:7.3f} ms | paged {ms_p:7.3f} ms | 间接寻址开销 {(ms_p/ms_c-1)*100:+5.1f}%")

# B. 容量收益(例 1 的结论画成图)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.loglog(kv_lens, t_contig, "o-", label="contiguous")
ax1.loglog(kv_lens, t_paged, "s-", label="paged")
ax1.set_xlabel("KV length"); ax1.set_ylabel("ms"); ax1.set_title("A. kernel: indirection overhead")
ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
ax2.bar(["naive prealloc", "paged"], [fn, fp], color=["tab:gray", "tab:green"])
ax2.set_ylabel("concurrent requests"); ax2.set_title("B. capacity under same memory (from Ex.1)")
plt.tight_layout(); plt.show()
```

### 实验结果解读

- **A 图**:paged 与连续版的耗时几乎重合(开销通常 <10%):页表常驻 L2、页内 16×64 fp16 = 2KB 连续读,合并访存未被破坏——第 05 篇"把随机粒度变粗"策略的胜利;
- **B 图**:同样显存,并发数提升数倍(例 1 的长尾负载下 ~5x)。**用 <10% 的 kernel 开销换数倍的并发=吞吐,这就是 PagedAttention 统治推理框架的原因**;
- 注意因果链:分页本身不让单次 attention 变快(甚至略慢),它让 **batch 变大**,而 batch 才是 decode 吞吐的杠杆(第 15 篇实验)。面试时把这条因果链讲对,比背"vLLM 用了分页"值钱得多。

---

## §5 练习 + 面试考点

### 动手练习

1. 给例 3 加上 **split-K**(第 17 篇):grid 扩成 `(BH, SPLITS)`,每段只扫自己的页区间,merge 复用第 17 篇公式——你将得到一个与 vLLM v1 decode kernel 结构等价的实现。
2. 实现**写入路径** `reshape_and_cache`:输入新 token 的 K/V 与各请求的写入位置,scatter 进池(查表 + `tl.store`)。对拍例 2 的 Python `append`。

### 面试高频考点

- **Q:PagedAttention 解决什么问题?原理?**
  A:KV cache 的显存碎片(内部:预分配未用;外部:变长分配缝隙)导致利用率 ~30%。借鉴 OS 虚拟内存:固定 16-token 物理块 + 每请求 block table 映射,按需分配,利用率 >90%;并发数(≈吞吐)等比提升。attention kernel 增加一层 block table 间接寻址,开销个位数百分比。
- **Q:block size 怎么选?**
  A:权衡:小块→碎片少、共享粒度细,但页表更大、gather 更碎、kernel 循环开销大;大块→访存更顺,但最后一块平均浪费 size/2、共享粒度粗。vLLM 默认 16,是在常见 head_dim 下对齐访存事务的经验值。
- **Q:两个请求共享相同 system prompt,paged 体系下怎么省显存?**
  A:它们的 block table 前若干项指向同一批物理块,引用计数管理;当某请求在共享前缀末尾继续生成(写入会污染共享块)时,copy-on-write 复制最后一块。系统化做这件事就是前缀缓存/RadixAttention(第 20 篇)。
- **Q:PagedAttention 和 FlashAttention 什么关系?**
  A:正交且叠加:Flash 解决"score 矩阵 IO"(算法层),Paged 解决"KV 存储管理"(内存管理层)。生产 kernel(vLLM)同时是 flash 式分块 + paged 式取数。
