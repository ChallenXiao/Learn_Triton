# Learn Triton — 从入门到精通(面向 AI Infra / 高性能计算 / 大模型推理训练加速岗位)

> 本文件是本项目的**总计划书 + 进度追踪表**。
> 任何 Agent 在本项目中工作时,必须先读完本文件,严格按照「文档格式规范」生成 notebook,
> 并在完成每一篇后**更新底部的 TODO 清单**(把 `[ ]` 改为 `[x]`,并填写完成日期)。

---

## 1. 项目目标

通过一套 29 篇的 Jupyter Notebook(`.ipynb`)教程,系统掌握:

1. **Triton GPU kernel 编程**:从零写出 softmax、GEMM、FlashAttention 等核心算子;
2. **大模型推理优化**:PagedAttention、RadixAttention、FlashAttention、Continuous Batching、量化,
   以及 vLLM / SGLang / TensorRT-LLM 三大框架的架构与 Triton 算子结合点;
3. **大模型训练优化**:算子融合(Liger-Kernel 风格)、混合精度、DDP / ZeRO(DeepSpeed)/ FSDP / TP / PP / CP;
4. **性能工程方法论**:roofline 分析、benchmark 规范、"该不该写 kernel"的决策能力。

最终能应对 AI infra / 推理加速岗位的面试:每篇 notebook 都标注对应的面试考点。

## 2. 运行环境约定

- **目标环境:Google Colab(免费档 T4 GPU,16GB 显存)**。学习者本机是 macOS,Triton 无法本地运行。
- 每个 notebook 的**第一个 cell 必须是统一的环境检测/安装代码**:
  - 检测是否有 CUDA GPU(`torch.cuda.is_available()`),打印 GPU 型号、显存、compute capability;
  - 若缺少依赖则 `pip install`(triton 随 PyTorch 2.x 自带,其余按需:`vllm`、`deepspeed` 等);
  - 若无 GPU,给出清晰的报错提示("请在 Colab 中选择 GPU 运行时")。
- **性能基准以 T4 为准**。T4 是 sm_75(Turing):
  - 支持 FP16 Tensor Core,**不支持 BF16 tl.dot、FP8、TMA、Hopper 特性**;
  - 涉及 FP8 / TMA / Hopper 的内容只做原理讲解并标注硬件要求,不作为必跑实验;
  - 所有实验的数据规模必须控制在 16GB 显存内。
- **分布式章节的特殊约定**(Colab 只有单卡):
  - 多卡通信类实验用「单卡数学等价模拟」或「CPU 多进程 + gloo 后端」实现闭环;
  - 显存收益类实验(ZeRO/FSDP)用「公式推导 + 单卡分阶段实测」闭环;
  - 必须诚实标注:哪些结论是单卡模拟得出的,真实多卡环境会有什么不同(通信开销、带宽瓶颈)。

## 3. 文档格式规范(每篇 notebook 必须遵守)

每篇 notebook 按以下五段式组织,使用 Markdown cell 分节:

1. **§1 是什么 & 能力边界**
   - 这个主题是什么、解决什么问题、在整个技术栈中的位置;
   - **能做什么 / 不能做什么**(能力边界必须明确写出,各至少 3 条);
2. **§2 递进式例子(2-4 个)**
   - 从最小可运行例子开始,逐步加变化(规模、边界条件、参数);
   - 每个例子的代码 cell 必须可独立运行,带正确性验证(`torch.testing.assert_close`);
3. **§3 知识连接**
   - 与**前面篇章**的联系(明确写"回顾第 NN 篇的 XXX");
   - 与**真实框架**的联系:PyTorch / CUDA / vLLM / SGLang / TensorRT-LLM / DeepSpeed 中
     对应的源码位置或等价实现(给出 GitHub 路径或 API 名);
4. **§4 闭环对比实验(必须有,不可省略)**
   - 用 `triton.testing.do_bench` 或 `torch.cuda.Event` 做规范计时;
   - **至少对比两种实现**(如:本篇 Triton kernel vs PyTorch 原生 vs torch.compile),
     输出表格 + matplotlib 性能曲线图;
   - 实验必须能让学习者**直观体会到不同技术的差别**(速度、显存、吞吐至少占其一);
   - 结尾用 2-3 句话解读实验结果:为什么快/慢,瓶颈在哪;
5. **§5 练习 + 面试考点**
   - 1-2 道动手练习题(给出提示,不给答案);
   - 列出本篇对应的高频面试题及答题要点。

**命名规范**:`notebooks/NN_主题英文名.ipynb`,如 `notebooks/01_pytorch_gpu_basics.ipynb`。

**质量验收标准**(Agent 自查):
- [ ] 所有代码 cell 从上到下顺序执行无错误(在有 GPU 的环境验证,或至少保证语法正确、API 真实存在);
- [ ] 正确性验证和性能对比实验齐全;
- [ ] 不杜撰 API:Triton API 以 triton-lang.org 官方文档为准,框架源码引用需真实存在;
- [ ] 中文讲解,代码注释可中英混合。

## 4. 课程大纲(6 阶段,29 篇)

### 阶段 0:地基(3 篇)— 补 PyTorch,建立 GPU 心智模型

| # | 文件名 | 主题 | 闭环对比实验 |
|---|--------|------|--------------|
| 01 | `01_pytorch_gpu_basics` | PyTorch Tensor 与 GPU 执行模型速成:tensor/device/dtype、异步执行与同步、`torch.cuda.Event` 计时、为什么需要自定义 kernel | 同一计算 CPU vs GPU、多次小 kernel vs 一次大 kernel 的耗时对比 |
| 02 | `02_gpu_architecture` | GPU 体系结构:SM、warp、HBM/L2/SRAM 存储层次、memory-bound vs compute-bound、roofline 模型 | 实测 T4 的带宽与 TFLOPS,绘制 roofline 图,标出几个常见算子的位置 |
| 03 | `03_hello_triton` | Triton 初识:在 CUDA 与 PyTorch 之间的定位、块级编程 vs 线程级编程、JIT 编译流程;能力边界(不适合:复杂控制流、动态形状高频变化、CPU 逻辑) | 第一个 vector-add kernel,与 `torch.add` 对比正确性与带宽利用率 |

### 阶段 1:Triton 编程模型(5 篇)— 语言核心

| # | 文件名 | 主题 | 闭环对比实验 |
|---|--------|------|--------------|
| 04 | `04_program_model` | grid / `tl.program_id` / `tl.arange` / mask 边界处理:Triton 的并行划分 | 不同 BLOCK_SIZE 下 vector-add 的性能曲线,解释为什么有最优点 |
| 05 | `05_memory_access` | 内存访问:`tl.load`/`tl.store`、stride、2D 索引、合并访存(coalescing) | 矩阵拷贝与转置:按行 vs 按列访问的带宽差异 |
| 06 | `06_fused_elementwise` | 逐元素融合 kernel:一次 load 做多件事(add+ReLU+dropout、GELU);算子融合为什么省带宽 | 融合 kernel vs PyTorch 链式调用:耗时 + 理论显存读写量对比 |
| 07 | `07_reduction` | 归约:`tl.sum`/`tl.max`、单 block 行内归约模式 | 第一版行 softmax vs `torch.softmax` |
| 08 | `08_benchmark_methodology` | 工程方法论:`triton.testing.do_bench`、`assert_close`、warmup/重复次数、`TRITON_INTERPRET=1` 调试 | 为前几篇 kernel 建立标准化 benchmark 报告模板(后续所有篇复用) |

### 阶段 2:核心算子(6 篇)— 面试主战场

| # | 文件名 | 主题 | 闭环对比实验 |
|---|--------|------|--------------|
| 09 | `09_softmax_online` | 数值稳定 softmax 与 online softmax(FlashAttention 的数学地基):为什么减 max、为什么能分块流式计算 | 朴素 vs 数值稳定 vs online 三版对比(正确性 + 性能) |
| 10 | `10_matmul_v1` | GEMM v1:tiling、`tl.dot`、accumulator,矩阵乘的块级写法 | 与 cuBLAS(`torch.matmul`)的 TFLOPS 对比,给出达成百分比 |
| 11 | `11_matmul_v2_autotune` | GEMM v2:`@triton.autotune`、num_warps/num_stages、L2 cache swizzle 分组 | 调优前后性能曲线;不同 M/N/K 形状下与 cuBLAS 的差距 |
| 12 | `12_layernorm_rmsnorm` | LayerNorm / RMSNorm kernel 与 GEMM epilogue 融合(bias+激活) | 与 PyTorch 原生、`torch.compile` 三方对比 |
| 13 | `13_backward_autograd` | 反向传播 kernel:`torch.autograd.Function` 封装、atomic 操作、LayerNorm backward | 端到端训练小 MLP:梯度正确性(gradcheck)+ 单步耗时对比 |
| 14 | `14_flash_attention_fwd` | FlashAttention 前向:tiling + online softmax 合体;IO 复杂度从 O(N²) 读写降到 O(N²/M) 的推导 | 朴素 attention vs 本篇实现 vs `F.scaled_dot_product_attention` 三方对比(速度 + 峰值显存,展示长序列下朴素版 OOM) |

### 阶段 3:大模型推理优化(8 篇)— 对接 vLLM / SGLang / TensorRT-LLM

| # | 文件名 | 主题 | 闭环对比实验 |
|---|--------|------|--------------|
| 15 | `15_llm_inference_anatomy` | 推理全景:prefill vs decode 两阶段、KV cache 原理与显存账本、decode 为什么是 memory-bound、TTFT/TPOT/吞吐指标定义 | 用 HuggingFace transformers 实测小模型(如 Qwen2.5-0.5B):prefill vs decode 的耗时结构、batch size 对吞吐的影响曲线 |
| 16 | `16_inference_kernels` | 推理高频小算子:RoPE、SwiGLU(SiLU+门控)、residual+RMSNorm 融合 | 对一个 Llama 风格 decoder block 做算子级替换,整 block 提速对比 |
| 17 | `17_flash_attention_advanced` | FlashAttention 进阶:causal mask、GQA/MQA、decode 阶段的 Flash-Decoding(split-K 归约);FA1/FA2/FA3 演进脉络 | prefill 形态 vs decode 形态的 attention 性能剖析;GQA 不同 group 数的影响 |
| 18 | `18_paged_attention` | **PagedAttention**:KV cache 碎片问题、分页存储 + block table 设计、vLLM 的核心思想;实现简化版 paged decode attention kernel(从 block table 间接寻址 gather KV) | 连续 KV vs 分页 KV 的 kernel 性能对比 + 显存碎片/利用率模拟实验(展示 paged 方案能容纳更多并发请求) |
| 19 | `19_continuous_batching` | **Continuous Batching**:static batching 的浪费、iteration-level 调度、与 chunked prefill 的配合;用 Python 写一个迷你调度器模拟器 | 模拟器实验:static vs continuous batching 在随机长度请求流下的吞吐与平均延迟对比(可视化 GPU 空泡) |
| 20 | `20_radix_attention` | **RadixAttention 与 prefix caching**(SGLang 核心):radix tree 管理共享前缀 KV、与 PagedAttention 的关系、适用场景(few-shot、多轮对话、并行采样) | 实现 radix tree 前缀匹配 + 共享 KV gather:有/无 prefix cache 的 TTFT 与显存占用对比(系统提示词复用场景) |
| 21 | `21_quantization_kernels` | 量化推理:对称/非对称量化、per-tensor/per-channel/per-group、INT8 quantize/dequantize kernel、weight-only W8A16 GEMM;GPTQ/AWQ/FP8 概念与硬件要求 | W8A16 GEMM vs FP16 GEMM:速度(decode 小 batch 场景)+ 精度误差实测;显存占用减半验证 |
| 22 | `22_inference_frameworks` | **推理框架对决:vLLM vs SGLang vs TensorRT-LLM**:架构图、调度/KV 管理/kernel 来源(Triton vs CUDA vs 闭源)对比、各自的 Triton 算子源码导读(vLLM `vllm/attention/ops/`、SGLang `sgl-kernel`)、选型决策树 | Colab 实测:vLLM vs HuggingFace transformers 跑同一小模型的吞吐对比(离线 batch 推理);TensorRT-LLM 因构建复杂仅做架构讲解并标注 |

### 阶段 4:大模型训练优化(5 篇)— 对接 DeepSpeed / FSDP / Megatron

| # | 文件名 | 主题 | 闭环对比实验 |
|---|--------|------|--------------|
| 23 | `23_training_fused_kernels` | 训练侧融合算子(Liger-Kernel 风格):fused cross-entropy(避免物化大 logits 梯度)、fused AdamW;与 apex/Liger 的对应 | 大 vocab 场景下 fused CE vs 朴素 CE:峰值显存 + 单步耗时对比 |
| 24 | `24_mixed_precision_memory` | 混合精度与显存解剖:fp16/bf16/AMP、loss scaling、训练显存四大件(参数/梯度/优化器状态/激活值)公式、activation checkpointing | `torch.cuda.memory` profiler 实测:fp32 vs AMP vs AMP+checkpointing 的显存与速度三方对比 |
| 25 | `25_data_parallel_zero` | 数据并行谱系:DDP(梯度 all-reduce)→ **ZeRO 1/2/3**(DeepSpeed,优化器状态/梯度/参数分片)→ **FSDP**;通信量与显存公式推导 | CPU 多进程 + gloo 实跑 2 进程 DDP 验证梯度同步;ZeRO 各 stage 显存公式推导 + 单卡用 DeepSpeed 实测 stage 0/1/2 的显存差异(若 Colab 环境允许) |
| 26 | `26_model_parallel_tp_pp_cp` | 模型并行:**TP**(Megatron 列切/行切 Linear、attention 头切分)、**PP**(GPipe/1F1B 气泡分析)、**CP**(序列并行/Ring Attention 思想);与 Triton 的关系(TP 切分后的 GEMM 形状变化对 kernel 调优的影响) | 单卡数学等价模拟:把 Linear 按列/行切成 2 份分别计算再拼接,验证与不切分结果一致 + 统计通信量;PP 气泡用时间线图模拟不同 micro-batch 数的气泡率 |
| 27 | `27_distributed_training_landscape` | 分布式训练全景与实战:DeepSpeed vs FSDP vs Megatron-LM 选型、3D 并行如何组合、训练框架中的 Triton 算子(Liger-Kernel 集成进 DeepSpeed/HF Trainer) | 用 Liger-Kernel 替换 HF 小模型的算子做一次真实微调:替换前后显存与吞吐对比 |

### 阶段 5:生态与收口(2 篇)

| # | 文件名 | 主题 | 闭环对比实验 |
|---|--------|------|--------------|
| 28 | `28_torch_compile_inductor` | Triton 与编译器生态:`torch.compile`/Inductor 如何自动生成 Triton kernel(`TORCH_LOGS=output_code` 读生成代码)、自定义 Triton 算子注册进 PyTorch(`torch.library`)、什么时候手写赢过编译器 | 同一融合模式:手写 Triton vs torch.compile 自动生成 vs eager 三方对比,并读懂生成的 kernel 代码 |
| 29 | `29_capstone_mini_gpt` | 综合项目:用本系列自写的 Triton kernel(RMSNorm/RoPE/FlashAttention/SwiGLU/W8A16 GEMM)组装 mini-GPT 推理 forward + **全系列面试高频问题清单与答题要点汇总** | 端到端:纯 PyTorch vs 自写 kernel 版的延迟/吞吐/显存对比;附完整面试题库(按篇索引) |

## 5. 生成流程约定(给执行 Agent)

1. **严格按编号顺序生成**,每篇生成前先重读本文件第 3 节格式规范;
2. 每篇生成后:
   - 在本文件第 6 节 TODO 清单中把对应项改为 `[x]` 并标注完成日期;
   - 在「备注」栏记录该篇的特殊事项(如:某实验需要 A100、某依赖版本锁定);
3. 学习者会在 Colab 实际运行验证,**若反馈报错,优先修复旧篇,再继续新篇**;
4. 涉及框架源码引用(vLLM/SGLang/DeepSpeed 等)时,必须给出真实的文件路径或 API 名,不确定时用 WebSearch/context7 查证,**禁止杜撰**;
5. 性能数字不要写死在 Markdown 里(不同 GPU 结果不同),让代码输出实测值。

## 6. TODO 清单(进度追踪 — 完成一篇更新一篇)

> 状态:`[ ]` 未开始 / `[~]` 进行中 / `[x]` 已完成(附日期)

### 阶段 0:地基
- [x] 01_pytorch_gpu_basics — PyTorch Tensor 与 GPU 执行模型(2026-06-09)
- [x] 02_gpu_architecture — GPU 体系结构与 roofline(2026-06-09)
- [x] 03_hello_triton — Triton 初识与第一个 kernel(2026-06-09)

### 阶段 1:Triton 编程模型
- [x] 04_program_model — grid/program_id/mask(2026-06-09)
- [x] 05_memory_access — load/store/stride/coalescing(2026-06-09)
- [ ] 06_fused_elementwise — 逐元素融合 kernel
- [ ] 07_reduction — 归约与第一版 softmax
- [ ] 08_benchmark_methodology — benchmark 与调试方法论

### 阶段 2:核心算子
- [ ] 09_softmax_online — 数值稳定与 online softmax
- [ ] 10_matmul_v1 — GEMM 基础版
- [ ] 11_matmul_v2_autotune — GEMM 调优版
- [ ] 12_layernorm_rmsnorm — Norm 类 kernel 与 epilogue 融合
- [ ] 13_backward_autograd — 反向 kernel 与 autograd 封装
- [ ] 14_flash_attention_fwd — FlashAttention 前向

### 阶段 3:大模型推理优化
- [ ] 15_llm_inference_anatomy — 推理全景与 KV cache
- [ ] 16_inference_kernels — RoPE/SwiGLU/融合 norm
- [ ] 17_flash_attention_advanced — causal/GQA/Flash-Decoding
- [ ] 18_paged_attention — PagedAttention 原理与简化实现
- [ ] 19_continuous_batching — 连续批处理与调度模拟器
- [ ] 20_radix_attention — RadixAttention 与 prefix caching
- [ ] 21_quantization_kernels — 量化 kernel 与 W8A16 GEMM
- [ ] 22_inference_frameworks — vLLM/SGLang/TensorRT-LLM 对决

### 阶段 4:大模型训练优化
- [ ] 23_training_fused_kernels — Liger 风格训练融合算子
- [ ] 24_mixed_precision_memory — 混合精度与显存解剖
- [ ] 25_data_parallel_zero — DDP/ZeRO/FSDP
- [ ] 26_model_parallel_tp_pp_cp — TP/PP/CP 模型并行
- [ ] 27_distributed_training_landscape — 分布式全景与 Liger 实战

### 阶段 5:生态与收口
- [ ] 28_torch_compile_inductor — torch.compile 与 Inductor
- [ ] 29_capstone_mini_gpt — 综合项目与面试题库

### 备注(各篇特殊事项记录区)
- (暂无)
