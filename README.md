# Learn Triton:从零到大模型推理/训练加速

一套面向 **AI Infra / 高性能计算 / 大模型推理训练加速岗位** 的 Triton GPU 编程系统教程,共 **6 个阶段、29 篇 Jupyter Notebook**,全部可在 **Google Colab 免费 T4 GPU** 上直接运行。

## 这套教程能学到什么

- **Triton kernel 编程**:从第一个 vector-add,到亲手实现 softmax、GEMM(逼近 cuBLAS)、LayerNorm/RMSNorm、FlashAttention;
- **大模型推理优化**:KV Cache、FlashAttention/Flash-Decoding、PagedAttention、RadixAttention、Continuous Batching、量化(W8A16),以及 vLLM / SGLang / TensorRT-LLM 三大框架的架构对比与源码导读;
- **大模型训练优化**:Liger-Kernel 风格融合算子、混合精度与显存解剖、DDP / ZeRO(DeepSpeed)/ FSDP / TP / PP / CP 分布式并行;
- **性能工程方法论**:roofline 分析、规范化 benchmark、读懂 torch.compile 生成的 Triton 代码、"该不该手写 kernel"的决策能力。

## 教程特色

每篇 notebook 都遵循统一的五段式结构:

1. **是什么 & 能力边界** —— 概念、作用、明确写出"能做什么 / 不能做什么";
2. **递进式例子** —— 2~4 个从最小可运行到逐步加难的例子,均带正确性验证;
3. **知识连接** —— 与前面篇章、与 PyTorch / vLLM / SGLang / DeepSpeed 等真实框架源码的对应关系;
4. **闭环对比实验** —— 每篇必有:至少两种实现的速度/显存/吞吐实测对比 + 图表 + 瓶颈解读;
5. **练习 + 面试考点** —— 动手题与高频面试题答题要点。

## 课程结构

| 阶段 | 篇目 | 内容 |
|------|------|------|
| **0 地基** | 01-03 | PyTorch GPU 执行模型、GPU 体系结构与 roofline、Triton 初识 |
| **1 Triton 编程模型** | 04-08 | grid/mask、内存访问与合并、算子融合、归约、benchmark 方法论 |
| **2 核心算子** | 09-14 | online softmax、GEMM 及 autotune、Norm 类算子、反向传播、FlashAttention |
| **3 大模型推理优化** | 15-22 | KV Cache、推理算子、Flash-Decoding、PagedAttention、Continuous Batching、RadixAttention、量化、vLLM/SGLang/TensorRT-LLM 对决 |
| **4 大模型训练优化** | 23-27 | 训练融合算子、混合精度、DDP/ZeRO/FSDP、TP/PP/CP、分布式全景与 Liger 实战 |
| **5 生态与收口** | 28-29 | torch.compile/Inductor、综合项目:自写 kernel 组装 mini-GPT + 面试题库 |

详细大纲与逐篇进度见 [CLAUDE.md](CLAUDE.md)。

## 如何使用

1. 打开 [Google Colab](https://colab.research.google.com/),上传 `notebooks/` 下的任意一篇(或通过 GitHub 链接直接打开);
2. 菜单选择 **代码执行程序 → 更改运行时类型 → T4 GPU**;
3. 从第一个 cell 顺序执行——首个 cell 会自动检测 GPU 并安装依赖;
4. 建议按编号顺序学习,后面的篇章会引用前面的结论。

> 本机为 macOS 也没关系:Triton 只能在 NVIDIA/AMD GPU 上运行,本教程的所有实验都以 Colab 免费 T4 为基准设计,显存控制在 16GB 以内。FP8 / TMA 等需要 Hopper 架构的特性会标注硬件要求、只做原理讲解。

## 仓库结构

```text
.
├── README.md          # 本文件:项目介绍
├── CLAUDE.md          # 总计划书 + 逐篇进度追踪(TODO 清单)
├── notebooks/         # 29 篇教程 notebook(.ipynb,核心内容)
├── sources/           # 各篇的 Markdown 源文件(便于 diff 与审阅)
└── tools/md2nb.py     # Markdown -> ipynb 转换脚本
```

## 学习进度

进度实时记录在 [CLAUDE.md](CLAUDE.md) 第 6 节 TODO 清单中。
