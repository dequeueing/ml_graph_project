# COMP5222 Group Project Proposal

## Revisiting Graph Reasoning Capabilities of Modern Large Language Models: A Replication and Extension Study

**中文题目：现代大语言模型图推理能力的重新评估：基于 GraphQA 的复现与扩展研究**

**Project Type:** Paper Reproduction & Extension  
**Course:** COMP5222 — Advanced Machine Learning with Graphs

---

## 1. Background and Motivation（研究背景与动机）

近年来，大语言模型（Large Language Models, LLMs）在自然语言理解、代码生成和复杂推理等任务上取得了显著进展。然而，图（Graph）是一种具有明确节点和边关系的结构化数据，LLM 能否准确理解和推理图结构，仍然是一个值得研究的问题。

2024 年，Google Research 的 Bahare Fatemi、Jonathan Halcrow 和 Bryan Perozzi 在 ICLR 发表了论文：

**Talk like a Graph: Encoding Graphs for Large Language Models**

- **Conference:** ICLR 2024
- **Paper:** https://arxiv.org/abs/2310.04560
- **Official Code:** https://github.com/google-research/talk-like-a-graph

这篇论文系统研究了如何将 Graph 编码成文本，让预训练 LLM 完成图推理任务。

作者构建了 **GraphQA Benchmark**，其中包含 Edge Existence、Node Degree、Node Count、Edge Count、Connected Nodes、Cycle Detection 等任务，并研究了三个关键因素：

1. **Graph Encoding**：不同的图文本表示方式如何影响推理准确率？
2. **Graph Task**：LLM 在不同类型的图推理任务上表现如何？
3. **Graph Structure**：不同的图拓扑结构如何影响 LLM 的推理能力？

原论文主要使用 Google 的 PaLM 62B 和不同规模的 PaLM 2 模型。实验发现，LLM 在一些基本图任务上的表现仍然有限，而 Graph Encoding、模型规模以及 Graph Topology 都会显著影响实验结果。

**然而，原论文研究的是 2023–2024 年的 LLM。此后，大语言模型在推理能力、上下文处理和指令遵循方面持续发展。**

因此，我们希望重新检验原论文的主要发现，研究现代 LLM 是否已经克服当时的局限，以及 Graph Encoding、Graph Complexity 和 Graph Representation 是否仍然影响其推理表现。

与此同时，我们希望引入 Token Consumption 和推理成本等评价指标，进一步分析 Graph Encoding 的效率。

## 2. Project Objectives（项目目标）

本项目属于 **Paper Reproduction & Extension**，不需要训练新的 LLM，也不计划提出新的 GNN 架构。

我们的目标是：

1. 复现 *Talk like a Graph* 中具有代表性的 Graph Reasoning 实验。
2. 在较新的 LLM 上重新评估原论文的核心结论。
3. 研究 Graph Complexity 和文本表示变化对推理结果的影响。
4. 从 Accuracy 和 Token Cost 两方面评价不同 Graph Encoding 方法。

通过上述工作，我们希望更系统地理解现代 LLM 在处理图结构化数据时的能力、局限与效率。

## 3. Research Questions（研究问题）

我们计划围绕以下五个 Research Questions 展开研究。

### RQ1 — Model Evolution

**Have modern LLMs improved their graph reasoning capabilities compared to earlier LLMs?**

*与早期 LLM 相比，现代 LLM 的图推理能力是否取得了实质性提升？*

原论文发现，当时的 LLM 即使面对简单图问题，也可能无法得到正确答案。

我们希望使用现代 LLM 重新执行具有代表性的 GraphQA 任务，并比较不同模型在准确率和错误类型上的差异。

除了整体性能提升，我们也希望分析这种进步是否均衡：现代模型是否只是在简单任务上取得进步，而在更复杂的结构推理任务上仍然存在局限？

为确保比较合理，我们将优先选择仍可访问的不同代际模型，在统一的输入与评价条件下进行比较。原论文中 PaLM 的结果将作为历史参考；如果无法重新调用原始模型，我们不会把跨论文的数值差异直接解释为严格的模型进步幅度。

### RQ2 — Graph Encoding

**Does graph encoding still matter for modern LLMs?**

*对于能力更强的现代 LLM，Graph Encoding 是否仍然显著影响图推理性能？*

原论文发现，相同的 Graph 使用不同文本编码方式，可能导致明显不同的推理结果。

例如：

- **Adjacency Encoding**：使用 `(A, B), (B, C)` 等边列表表示图。
- **Incident Encoding**：使用 `Node B is connected to A and C` 等描述表示节点与邻居之间的关系。

我们计划比较不同编码方式在现代 LLM 上的表现，并分析是否存在某一种编码始终优于其他方式。

进一步地，我们希望检验 Graph Encoding 的相对优势是否会随模型能力增强而减小。

### RQ4 — Graph Complexity

**How does graph complexity affect modern LLMs?**

*图结构的复杂度如何影响现代 LLM 的推理准确率？*

我们将重点研究：

- **Graph Size**：节点数量增加是否导致性能下降？
- **Graph Density**：边的密度变化如何影响推理？
- **Graph Topology**：Path、Star、Random Graph 等不同结构是否具有不同难度？

我们希望发现不同模型在图复杂度增加时的性能变化规律，并研究更强的 LLM 是否能够处理更复杂的图。

### RQ5 — Representation Robustness

**How robust are LLMs to graph representation changes?**

*当图的语义结构保持不变，仅修改其文本表示方式时，LLM 的推理结果是否稳定？*

对于同一个 Graph，我们可以：

- 重新排列 Edge List 的顺序。
- 改变节点的命名方式，例如字母、数字或人名。
- 改变 Neighbor List 中邻居的排列顺序。

这些变化不会改变 Graph 本身，因此正确的图推理结果也不应该发生变化。

我们将比较 LLM 在这些等价表示下的 Accuracy 和 Prediction Consistency，以评估其对图文本表示变化的敏感程度。

### RQ6 — Accuracy–Cost Trade-off

**What is the accuracy–token cost trade-off of different graph encodings?**

*不同 Graph Encoding 方法在推理准确率与 Token 成本之间存在怎样的权衡？*

不同编码方式可能具有不同的文本长度。

例如，Incident Encoding 可能重复描述某些边，从而增加输入 Token 数量；但这种冗余也可能帮助 LLM 更准确地理解图结构。

因此，我们计划同时记录：

- **Accuracy**：推理答案的正确率。
- **Input Tokens**：图表示与问题的输入 Token 数量。
- **Output Tokens**：模型回答所使用的 Token 数量。
- **Inference Cost**：在可获得计费信息时计算 API 调用成本。

我们希望分析，准确率较高的 Graph Encoding 是否一定具有更高的 Token 成本，以及不同方法的 Accuracy–Cost Trade-off。

*注：沿用前期讨论的 RQ 编号。原 RQ3 关于 LLM 是否“真正理解”Graph 的问题暂不纳入本项目，因为该问题较难通过有限的实验直接验证。*

## 4. Experimental Methodology（实验方法）

### 4.1 Benchmark and Graph Tasks

我们计划复用原论文的 GraphQA Benchmark 和公开的图生成代码，并选择以下代表性任务：

| Task | Description |
|---|---|
| Edge Existence | 判断两个节点之间是否存在边 |
| Node Degree | 计算指定节点的度数 |
| Cycle Detection | 判断图中是否存在环 |

这三个任务分别覆盖基础关系识别、局部结构统计和全局结构判断。

我们将使用 NetworkX 生成 Graph，并通过标准图算法计算 Ground Truth，使实验结果能够自动验证。

### 4.2 LLM Selection

我们计划选择至少两个不同代际或能力水平的 LLM，其中至少一个是现代模型。

所有模型使用相同的数据集、Graph Encoding 和问题描述，以尽可能保证实验公平。

实验不进行 Fine-tuning，而是通过 LLM API 调用模型进行推理。模型版本将在最终报告中明确记录。

### 4.3 Experimental Design

实验分为三个部分。

**Part I — Reproduction（RQ1、RQ2）**

首先，复现原论文关于 Graph Encoding 和 Graph Reasoning 的核心实验。

在相同 Graph 和任务上，比较不同模型与不同编码方式的 Accuracy，并将结果与原论文进行对照。

**Part II — Structural Analysis（RQ4、RQ5）**

在第一部分基础上，控制其他变量，研究 Graph Size、Density、Topology 对性能的影响。

随后，对同一批 Graph 进行不改变其数学结构的文本表示变换，测试模型预测是否稳定。

**Part III — Efficiency Analysis（RQ6）**

复用前两部分实验得到的 API 调用记录，统计不同 Graph Encoding 的 Input Tokens、Output Tokens 和 Accuracy。

通过比较 Token Consumption 与推理结果，分析不同方法的性能与效率权衡。

### 4.4 Experimental Controls

为提高实验结果的可信度，我们计划采用以下控制方法：

1. **Paired Evaluation**：不同模型和 Encoding 使用相同的测试图。
2. **Controlled Prompting**：尽量固定 Prompt Template、输出格式以及模型推理配置。
3. **Balanced Evaluation**：对于 Edge Existence、Cycle Detection 等二分类任务，控制正负样本比例，并报告 Majority Baseline，避免答案分布造成误导。
4. **Reproducibility**：固定随机种子，记录模型版本、Graph、Prompt、Ground Truth 和模型输出。
5. **Statistical Analysis**：在样本量允许的情况下，报告 Accuracy 的置信区间，并分析模型间的差异。

## 5. Implementation and Feasibility（实现与可行性）

本项目的一个重要目标是保持较低的实现成本。

**Hardware Requirements**

本地计算部分使用 CPU 即可完成，包括 Graph Generation、Graph Encoding、Ground Truth Computation 和 Result Analysis。

LLM 推理通过在线 API 完成，不需要本地部署模型，也不需要 GPU。

**Implementation**

主要使用 Python，并计划复用：

- NetworkX：图生成与图算法。
- 原论文官方代码：GraphQA Tasks 和 Graph Encoding。
- LLM API：调用模型完成图推理。
- Pandas / Matplotlib：结果统计与可视化。

**Experimental Scale**

为了控制成本，我们计划先使用少量图样本完成 Pilot Experiments，再逐步扩展。

初步计划在核心实验中采用两个模型、两种主要 Graph Encoding 和三个 Graph Tasks。复杂度与鲁棒性实验将分别选择少量代表性配置，而不执行所有变量组合的完整笛卡尔积。

我们希望将单次核心实验批次的运行时间控制在约一小时以内，但实际规模将根据 API 响应速度、速率限制与费用预算调整。

这项研究不要求完整复现原论文全部模型、Prompting 方法和 Graph Topology，而是重点复现其核心发现，并进行有针对性的扩展。

## 6. Expected Outcomes（预期成果）

我们预计完成以下成果：

**1. Reproduction Results**

在现代 LLM 上重新评估 GraphQA 的代表性任务，并分析与原论文结论之间的一致性及差异。

**2. Experimental Findings**

围绕五个 RQ，总结以下问题：

- 现代 LLM 的 Graph Reasoning 能力是否有所改善？
- Graph Encoding 的影响是否依然明显？
- Graph Complexity 如何影响推理性能？
- 等价的 Graph 表示是否产生一致的答案？
- 不同 Graph Encoding 是否存在明显的 Accuracy–Cost Trade-off？

**3. Reproducible Artifact**

提供可以复用的实验代码，包括 Graph Generator、Text Encoder、LLM Evaluation Runner、Result Analyzer，以及必要的配置与说明。

**4. Final Report and Presentation**

最终报告将总结实验设置、主要结果、局限与研究发现，并通过图表展示不同模型及编码方法之间的差异。

本项目不预设现代 LLM 一定优于历史模型，也不预设某种 Encoding 必然更好。无论实验结果是否与原论文一致，都将通过分析解释观察到的现象。

## 7. Project Schedule（项目计划）

| 时间 | 工作内容 |
|---|---|
| Oct 28–31 | 确定研究范围，提交 Project Proposal |
| Nov 1–7 | 搭建 GraphQA Benchmark 和 LLM Evaluation Runner |
| Nov 8–16 | 完成 RQ1、RQ2 的主要复现实验 |
| Nov 17–24 | 完成 RQ4、RQ5、RQ6 的扩展实验 |
| Nov 25–30 | 整理结果、分析发现并撰写报告 |
| Dec 1 | 提交 Project Code and Report |
| Dec 2 | 提交 Project Slides |

## 8. Conclusion（总结）

本项目以 ICLR 2024 论文 *Talk like a Graph: Encoding Graphs for Large Language Models* 为基础，采用 Reproduction & Extension 的形式，重新评估现代 LLM 的图推理能力。

我们将从 **Model Evolution、Graph Encoding、Graph Complexity、Representation Robustness 和 Accuracy–Cost Trade-off** 五个维度展开实验研究。

本项目的主要贡献不是开发新的图学习模型，而是通过可控、可复现的实验，验证已有研究发现是否仍然适用于现代 LLM，并进一步分析模型能力、图结构、文本表示和推理成本之间的关系。

最终，我们希望得到一组具有解释性且可复现的实验发现，为使用 LLM 处理结构化图数据提供参考。

## References

[1] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. *Talk like a Graph: Encoding Graphs for Large Language Models.* International Conference on Learning Representations (ICLR), 2024. https://arxiv.org/abs/2310.04560

[2] Google Research. *Talk Like a Graph — Official Implementation.* https://github.com/google-research/talk-like-a-graph
