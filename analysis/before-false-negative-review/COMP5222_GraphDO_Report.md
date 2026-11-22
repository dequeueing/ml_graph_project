# Revisiting Edge Ordering in Graph Reasoning with Large Language Models

**Taojie Wang, Chiyu Hao, and Zhengdong Huang**  
COMP5222 Advanced Machine Learning with Graphs, Hong Kong University of Science and Technology

## Abstract

Large language models often receive graphs as text. GraphDO reports that structured edge ordering improves accuracy and that the most useful order depends on the task. We revisit these findings at half scale with the public implementation, keeping six tasks, five edge orders, and the original prompt styles. We evaluate 17,500 traditional-task responses per model from DeepSeek V4.1 Flash, GLM-5.3, and GPT-6 Luna, and 2,250 node-classification responses per model from DeepSeek and Luna. The traditional-task scores average 92.79%, 90.86%, and 83.77%, respectively. Ordering gains over random order are small, and all 300 paired comparisons have Holm-adjusted p-values above 0.05. Hamiltonian path is the most sensitive task for DeepSeek and Luna, while weighted shortest path is most sensitive for GLM. In node classification, PPR raises DeepSeek's accuracy by 2.44 percentage points and lowers Luna's by 10.22 points. The results show that the effect of edge order varies with the task and model.

## 1. Introduction

Graphs represent relationships such as dependencies between tasks and citations between papers. Large language models (LLMs) often receive graphs as text, and how best to present these relationships for reasoning remains an open question. Ge et al. (2025) investigate one aspect of this problem with GraphDO. They find that presenting edges in a structured order improves GPT-3.5's accuracy over a randomly ordered edge list, and that the order yielding the highest accuracy depends on the task. [Ge et al. (2025), §§4–5](https://aclanthology.org/2025.acl-long.321/)

In this study, we replicate the main GraphDO experiment with the public implementation. We test whether the findings hold for three newer models, DeepSeek V4.1 Flash, GLM-5.3, and GPT-6 Luna, on connectivity, cycle detection, weighted shortest path, Hamiltonian path, topological sorting, and node classification. As in GraphDO, we compare random order with breadth-first search (BFS), depth-first search (DFS), PageRank (PR), and Personalized PageRank (PPR). We also vary the prompt style, which specifies the instructions and examples that accompany each question (§2.2). The first five tasks use zero-shot, zero-shot chain-of-thought (CoT), few-shot, CoT with worked examples, and CoT-BAG. Node classification uses zero-shot prompts. Following Ge et al., we ask three research questions.

1. **RQ1.** How does each structured edge order change accuracy relative to random order?
2. **RQ2.** Which graph tasks are most sensitive to edge order for each model and prompt style?
3. **RQ3.** Which orders score highest on each task, and how do the rankings vary by model?

The replication recovers GraphDO's finding that edge order matters more for some tasks than others. Connectivity and cycle detection are consistently less sensitive than the three tasks that require a vertex sequence. Within those three tasks, the highest sensitivity depends on the model and prompt style. The best order also varies. For example, PPR leads on weighted shortest path for DeepSeek and Luna, while random order leads for GLM. Across the five traditional tasks, the mean gains are small and none of the paired comparisons reaches statistical significance after Holm correction.

## 2. Background

### 2.1 Graph prompts

A graph $G=(V,E)$ consists of vertices $V$ and edges $E$. A graph reasoning prompt contains a textual description of $G$ and a question $q$. The prompt style $p$ specifies which instructions or worked examples accompany the question. For an edge-list encoding, let $o$ be the rule that orders the edges and $g(G,o)$ the resulting text. A language model $M$ takes the prompt style, graph description, and question as inputs and returns $\hat{y}_o$.

$$
\hat{y}_o = M\bigl(p, g(G,o), q\bigr).
$$

GraphDO places the edges in a fixed text template. For example, the lists `[(0, 1), (1, 2), (0, 2)]` and `[(0, 2), (0, 1), (1, 2)]` describe the same undirected triangle, so Yes is the correct answer to a cycle-detection question for both inputs. By keeping $G$, $q$, and $p$ fixed, GraphDO can compare accuracy across the two descriptions. [Ge et al. (2025), §2.1 and §3.1](https://aclanthology.org/2025.acl-long.321.pdf#page=2)

### 2.2 Prompt styles

The same graph question can be presented with different amounts of guidance. For example, a cycle-detection prompt can present the question by itself, ask the model to reason through the graph, or show how a similar question was solved on another graph. Each version can use the same edge sequence for the graph being tested.

GraphDO uses five prompt styles for traditional graph tasks. The table below describes each style and gives an example prompt. CoT stands for chain-of-thought, and BAG stands for Build-a-Graph. [Ge et al. (2025), Appendix A.1 and Table 3](https://aclanthology.org/2025.acl-long.321.pdf#page=14)

The examples ask whether the same triangle contains a cycle. For styles that use demonstrations, one solved question on a two-edge path is shown before the triangle question.

| Prompt style | What the model receives | Example |
|---|---|---|
| Zero-shot | The test graph and its question | `Graph: [(0, 1), (1, 2), (0, 2)]. Is there a cycle?` |
| Zero-shot CoT | The test graph and question with an instruction to reason step by step | `Graph: [(0, 1), (1, 2), (0, 2)]. Is there a cycle? Think step by step.` |
| Few-shot | Example graphs, questions, and final answers before the test question | `Example graph: [(0, 1), (1, 2)]. Is there a cycle? Answer: No.` Then ask the triangle question. |
| CoT | Worked examples that show the reasoning leading to each answer | `Example graph: [(0, 1), (1, 2)]. Is there a cycle? These edges form a path with no loop. Answer: No.` Then ask the triangle question. |
| CoT-BAG | The CoT examples and an instruction to reconstruct the test graph from its nodes and edges before answering | Give the worked CoT example, then ask the triangle question with the instruction `Construct the graph from its nodes and edges first.` |

### 2.3 Graph reasoning and node classification

Graph tasks differ in the information they provide and the answers they require. A structural question uses the edges and any associated weights to ask whether a property holds or to request a solution that satisfies a constraint. For task like node classification, labels $\ell(v)$ are observed for vertices $v$ in a subset $L\subseteq V$. The goal is to predict $\ell(u)$ for a target vertex $u\in V\setminus L$ using the graph and the observed labels.

**Table 1. GraphDO tasks and expected answers.** [Ge et al. (2025), §2.2](https://aclanthology.org/2025.acl-long.321.pdf#page=3)

| Task | Meaning | Input graph | Expected answer |
|---|---|---|---|
| Connectivity | Check whether one queried vertex can reach the other along graph edges. | Undirected graph with two query vertices | Yes or No |
| Cycle detection | Check whether the graph contains a closed loop of edges. | Undirected graph | Yes or No |
| Hamiltonian path | Check whether a single path can visit every vertex exactly once. | Undirected graph | Yes or No. If Yes, give the path as a vertex sequence. |
| Weighted shortest path | Find the route with the lowest total edge weight from the source to the target. | Weighted undirected graph with a source and a target | A vertex sequence from source to target |
| Topological sorting | Put all vertices in an order that respects the dependencies encoded by the directed edges. | Directed acyclic graph | A list of all vertices with $u$ before $v$ for each edge $(u,v)$ |
| Node classification | Infer a vertex's class from the graph connections and the known vertex labels. | Citation subgraph with one masked label at vertex $u$, so $L=V\setminus\{u\}$ | The missing label $\ell(u)$ |

### 2.4 Edge ordering strategies

An ordering rule $o$ turns an edge set $E=\{e_1,\ldots,e_m\}$ into a sequence $L_o=(e_{\pi(1)},\ldots,e_{\pi(m)})$, where $\pi$ is a permutation of the edge indices. The sequence determines which edges appear first and which ones appear together in the prompt. For example, a traversal-based rule orders edges as a search encounters them while a ranking-based rule places edges incident to higher-ranked vertices earlier. 

GraphDO compares five edge orders, including a random baseline. Table 2 describes each ordering rule and gives the edge sequence it produces for the graph in Figure 1.

![Five-node undirected path from 3 through 1, 0, and 2 to 4](</Users/taojiewang/Downloads/COMP5222 (L1) - Advanced Machine Learning with Graphs/figures/graphdo-edge-order-example.png>)

**Figure 1. Undirected graph used in the edge-order examples.**

**Table 2. Edge orders in GraphDO.** BFS and DFS start at vertex 0. PPR uses the node-classification rule with vertex 4 as the target. [Ge et al. (2025), §3.2](https://aclanthology.org/2025.acl-long.321.pdf#page=3)

| Edge order | How it arranges edges | Example |
|---|---|---|
| Random (baseline) | Shuffles the full edge list. | `[(1, 3), (0, 2), (2, 4), (0, 1)]` |
| Breadth-first search (BFS) | Visits adjacent edges level by level. | `[(0, 1), (0, 2), (1, 3), (2, 4)]` |
| Depth-first search (DFS) | Follows a branch of adjacent edges as far as possible before backtracking. | `[(0, 1), (0, 2), (2, 4), (1, 3)]` |
| PageRank (PR) | Ranks vertices by PageRank and lists edges incident to higher-ranked vertices earlier. | `[(0, 1), (1, 3), (0, 2), (2, 4)]` |
| Personalized PageRank (PPR) | Uses task-specific information to personalize the vertex ranking and lists edges incident to higher-ranked vertices earlier. | `[(0, 2), (2, 4), (0, 1), (1, 3)]` |

## 3. Experimental Design

### 3.1 Tasks and samples

Table 1 lists the six tasks we evaluate, all drawn from Ge et al. (2025). We use the graph generator, question templates, demonstrations, and evaluator from the public GraphDO implementation at commit `4ba8f925`. [GraphDO implementation](https://github.com/YuyaoGe/GraphDO/tree/4ba8f925c76574e2fca8d4f32f753b57a341b44f)

For the five traditional tasks, we use seed 42 to generate 140 Erdős–Rényi graphs per task. The generator uses an edge probability of 0.3 and samples graphs with 5 to 15 vertices. It caps the graph size at 12 vertices for the Hamiltonian-path task. For weighted shortest path, it assigns positive integer weights and chooses a source and target connected by a path. For topological sorting, it adds only edges that point forward in a sampled vertex ordering to ensure the graph is acyclic. [GraphDO Appendix B](https://aclanthology.org/2025.acl-long.321.pdf#page=14) · [Pinned graph generator](https://github.com/YuyaoGe/GraphDO/blob/4ba8f925c76574e2fca8d4f32f753b57a341b44f/graph_generator.py)

For node classification, we use the Cora, CiteSeer, and PubMed citation networks provided by Planetoid. We draw 50-vertex subgraphs using 3-hop ego sampling and Forest Fire sampling. The Forest Fire sampler uses a burn probability of 0.3. We draw 25 subgraphs for each combination of dataset and sampler. In each subgraph, we sample one target vertex and mask its label. [Planetoid citation datasets](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html) · [GraphDO Appendix B](https://aclanthology.org/2025.acl-long.321.pdf#page=14)

To stay within our token budget, we use half the original number of graphs per setting. For each task, we reuse the same graph-question pairs across edge orders, prompt styles, and models.

**Table 3. Sample sizes and prompting conditions.**

| Task | Query | Original sample | Replication sample | Prompt styles |
|---|---|---:|---:|---|
| Connectivity | Determine whether two queried vertices are connected | 280 | 140 | Zero-shot, zero-shot CoT, few-shot, CoT, CoT-BAG |
| Cycle detection | Determine whether an undirected graph contains a cycle | 280 | 140 | Zero-shot, zero-shot CoT, few-shot, CoT, CoT-BAG |
| Weighted shortest path | Return a minimum-weight path between two vertices | 280 | 140 | Zero-shot, zero-shot CoT, few-shot, CoT, CoT-BAG |
| Hamiltonian path | Determine whether a path visits every vertex once and give the path if it exists | 280 | 140 | Zero-shot, zero-shot CoT, few-shot, CoT, CoT-BAG |
| Topological sorting | Return a vertex ordering consistent with the directed edges | 280 | 140 | Zero-shot, zero-shot CoT, few-shot, CoT, CoT-BAG |
| Node classification | Predict one hidden vertex label in a labeled subgraph | 50 per dataset and sampler | 25 per dataset and sampler | Zero-shot |

### 3.2 Edge orders and prompts

For each of the five orders in Table 2, we generate one edge list per sampled graph using the settings in Table 4. We reuse each list across prompt styles, models, and repeats.

**Table 4. Edge-order settings used in the replication.** [GraphDO Appendix A.2](https://aclanthology.org/2025.acl-long.321.pdf#page=13) · [Pinned ordering implementation](https://github.com/YuyaoGe/GraphDO/blob/4ba8f925c76574e2fca8d4f32f753b57a341b44f/graph_ordering.py) · [Pinned description generator](https://github.com/YuyaoGe/GraphDO/blob/4ba8f925c76574e2fca8d4f32f753b57a341b44f/description_generator.py)

| Order | Applies to | Settings used |
|---|---|---|
| Random | All tasks | Initialize Python's random number generator with seed 42 at the start of benchmark construction, then shuffle each graph's edge list once. |
| BFS and DFS | All tasks | Traverse the line graph, where edges sharing an endpoint are adjacent. Start at the first edge incident to the first stored vertex, or the first listed edge if that vertex is isolated. Resume at the first unvisited edge when a component is exhausted. |
| PR and PPR | All tasks | Use NetworkX PageRank with a damping factor of $\alpha=0.85$. For the weighted shortest-path task, the ranking uses the stored edge weights. |
| PPR | Connectivity | Split the personalization weight equally between the two queried vertices. |
| PPR | Cycle detection | Split the weight equally among the vertices of one computed cycle. Use a uniform prior for an acyclic graph. |
| PPR | Hamiltonian path | Use the released implementation's uniform prior over all vertices. |
| PPR | Weighted shortest path | Split the weight equally among the vertices of a computed minimum-weight path between the queried endpoints. |
| PPR | Topological sorting | Split the weight equally among vertices with in-degree zero. |
| PPR | Node classification | Assign each vertex $v$ reachable from the target $u$ a weight of $d_{\max}-d(u,v)+1$, where $d_{\max}$ is the maximum distance from $u$ to a reachable vertex. Assign $10^{-5}$ to the remaining vertices and normalize the weights. |

We convert the five edge lists into text using the same template for a given task. We check that each description preserves the original edge set, including edge direction for directed graphs. An undirected edge can be written with either endpoint first. The encoder retains the edge weights for shortest path and, for node classification, appends the observed labels with the same target label masked in every description. [Pinned text encoder](https://github.com/YuyaoGe/GraphDO/blob/4ba8f925c76574e2fca8d4f32f753b57a341b44f/text_encoder.py)

We use the five prompt styles defined in §2.2 for each traditional task and zero-shot for node classification, as listed in Table 3. Within a prompt style, the question, instructions, and any demonstrations stay fixed across all five edge orders. [GraphDO Appendix A.1 and Table 3](https://aclanthology.org/2025.acl-long.321.pdf#page=14)

### 3.3 Model configurations

GraphDO used GPT-3.5-Turbo-0613 for its main experiments. Its supplementary experiments evaluated Llama 2 7B Chat, Llama 2 13B Chat, Qwen2-7B, Mistral-7B, and Vicuna-7B-v1.5. OpenAI retired the `gpt-3.5-turbo-0613` API snapshot on September 13, 2024, so we run the replication with DeepSeek V4.1 Flash, GLM-5.3, and GPT-6 Luna. These three models also let us test whether the reported edge-order effects carry over to newer LLMs. [Ge et al. (2025), §4.1](https://aclanthology.org/2025.acl-long.321.pdf#page=4) · [OpenAI API deprecations](https://developers.openai.com/api/docs/deprecations#past-deprecations)


### 3.4 Scoring and analysis

#### Final scoring

We score answers against the ground truth for each graph. For connectivity and cycle detection, we compare an explicit Yes or No with the graph's ground truth. For weighted shortest path, we check the endpoints, edges, and total weight of the returned path. For Hamiltonian path, we validate an explicit path or check a negative answer using an exact search. For topological sorting, we check the precedence constraints in the edge list and allow the omission of isolated vertices that the prompt does not name. We identify the stated answer from its wording and position, including labeled conclusions followed by explanations. Candidate solutions and edge-verification lists are not treated as final answers. For traditional tasks, the scoring procedure uses the pinned GraphDO evaluator as a fallback when it cannot extract an explicit answer. For node classification, we extract the predicted integer label and compare it with the masked label in the dataset. [Pinned GraphDO evaluator](https://github.com/YuyaoGe/GraphDO/blob/4ba8f925c76574e2fca8d4f32f753b57a341b44f/evaluator.py)

For an experimental cell with $n$ responses, the accuracy in percent is $A=\frac{100}{n}\sum_{i=1}^{n}s_i$, where $s_i$ is 1 if the answer is correct and 0 otherwise. Each traditional-task cell fixes the model, task, prompt style, and edge order, with one response per graph and 140 responses in total. The final traditional-task data contain 17,500 responses per model, including 5,991 GLM responses generated with a higher output cap.

Each node-classification cell fixes the model, dataset, sampler, and edge order. There are 25 graph questions with three runs per question, giving 75 responses. The reported ordering comparisons use DeepSeek and Luna, which returned a visible answer to every node-classification prompt (§6).

#### Analysis by research question

**Table 5. Comparisons and metrics for the three research questions.** [GraphDO Figure 4](https://aclanthology.org/2025.acl-long.321.pdf#page=6) · [GraphDO Figure 5](https://aclanthology.org/2025.acl-long.321.pdf#page=7)

| Research question | Comparison | Metric |
|---|---|---|
| RQ1. Change from random order | Pair each structured-order response with its random-order counterpart by graph ID. For node classification, pair by repeat index as well. | Difference in accuracy in percentage points. Relative change expresses that difference as a percentage of the random-order accuracy when the baseline is nonzero. |
| RQ2. Sensitivity to order | For each model and traditional task, compare the five orders within each prompt style. | Population variance of the five accuracy values. Average the five style-specific variances to summarize each model and task. |
| RQ3. Highest-scoring order | For each model and traditional task, average each order's accuracy across the five prompt styles. For node classification, compare orders within each dataset-sampler group and summarize across groups. | Rank orders by accuracy, retaining ties. Compare orders using the relative difference in their scores. |

For each model, traditional task, and prompt style, we use a two-sided exact McNemar test to compare each structured order with random order on the same 140 graph questions. The three models, five tasks, five prompt styles, and four structured orders give 300 comparisons. We apply Holm correction across these 300 tests. [Fagerland et al. (2013)](https://doi.org/10.1186/1471-2288-13-91) · [Holm (1979)](https://doi.org/10.2307/4615733)

We report the 48 node-classification comparisons descriptively because repeats share prompts and subgraphs can overlap.

## 4. Results

### 4.1 Research questions and previous results

Table 6 summarizes the three research questions and main findings from the experiments by Ge et al. (2025) with GPT-3.5-Turbo-0613. Across these experiments, structured edge orders consistently improved accuracy over random order. The highest-scoring order varied by task. Accuracy also varied more across orders on tasks that require global reasoning, such as topological sorting and weighted shortest path, than on local reasoning tasks such as connectivity and cycle detection. In the following subsections, we revisit each question with DeepSeek V4.1 Flash, GLM-5.3, and GPT-6 Luna. [Ge et al. (2025), §§2.2 and 4.2](https://aclanthology.org/2025.acl-long.321.pdf)

**Table 6. Research questions and findings in the original GraphDO study.**

| Research question | Original GraphDO finding |
|---|---|
| **RQ1.** How does edge order affect accuracy compared with random order? | Structured orders outperformed random order in all 100 traditional-task comparisons. For example, BFS increased mean cycle-detection accuracy from 64.50% to 72.71%. Structured orders also improved node classification across all six dataset-sampler settings. [GraphDO Table 1](https://aclanthology.org/2025.acl-long.321.pdf#page=5) · [GraphDO Table 2](https://aclanthology.org/2025.acl-long.321.pdf#page=6) |
| **RQ2.** Which graph tasks are most sensitive to edge order? | Accuracy varied less across orders for connectivity and cycle detection than for Hamiltonian path, topological sorting, and weighted shortest path. Weighted shortest path had the largest mean variance across the five prompt styles. [GraphDO Figure 4](https://aclanthology.org/2025.acl-long.321.pdf#page=6) |
| **RQ3.** Which edge order performs best on each task? | Averaged across prompt styles, BFS ranked first on connectivity, cycle detection, and weighted shortest path. DFS ranked first on Hamiltonian path and topological sorting. PageRank led or tied for first in all six node-classification settings. [GraphDO Table 1](https://aclanthology.org/2025.acl-long.321.pdf#page=5) · [GraphDO Table 2](https://aclanthology.org/2025.acl-long.321.pdf#page=6) |

### 4.2 RQ1. How much does edge order affect accuracy?

> **Finding 1.** Average accuracy gains from structured edge ordering are substantially smaller than in GraphDO.

The original study reported higher mean accuracy under all four structured orders, whereas our models' mean accuracies varied much less across orders in Figure 2. DeepSeek's mean accuracy remained within 0.23 percentage points of random order. More strikingly, GLM reversed the original pattern: all four structured orders reduced its mean accuracy, by 0.63–1.20 percentage points. Even Luna's strongest gain, 2.49 points under PPR, was less than one-fifth of the original study's largest gain of 12.56 points under DFS. [GraphDO Table 1](https://aclanthology.org/2025.acl-long.321.pdf#page=5)

![Mean traditional-task accuracy by edge order for the original GPT-3.5 study and our three replication models](</Users/taojiewang/Downloads/COMP5222 (L1) - Advanced Machine Learning with Graphs/figures/graphdo-audited-accuracy.png>)

**Figure 2. Mean accuracy across edge orders on traditional graph tasks.** Each bar averages accuracy across five tasks and five prompt styles. GPT-3.5-Turbo-0613 values come from [GraphDO Table 1](https://aclanthology.org/2025.acl-long.321.pdf#page=5). The other models use our final scores. Gray bars indicate the random-order baseline.

> **Finding 2.** Improvements and declines are nearly balanced across traditional-task comparisons.

GraphDO reported higher accuracy under structured ordering in all 100 traditional-task comparisons. In our replication, the breakdown in Figure 3 includes 106 comparisons with higher accuracy, 95 with lower accuracy, and 99 ties. Among the 201 comparisons with a change, only 52.7% favored structured ordering. This near balance masks differences between models: Luna and DeepSeek had more improvements than declines, whereas GLM had more than twice as many declines as improvements, with 41 versus 19.

![Counts of higher, equal, and lower accuracy relative to random order for the original study and each replication model](</Users/taojiewang/Downloads/COMP5222 (L1) - Advanced Machine Learning with Graphs/figures/graphdo-audited-consistency.png>)

**Figure 3. Consistency of accuracy changes relative to random order.** Each bar contains 100 comparisons: five traditional tasks × five prompt styles × four structured orders. Segment labels count comparisons with higher, equal, or lower accuracy than the corresponding random-order baseline. The categories describe numerical accuracy differences. Original-study counts are derived from [GraphDO Table 1](https://aclanthology.org/2025.acl-long.321.pdf#page=5).

**Table 7. Average accuracy changes relative to random order.** Random is accuracy in percent. The four structured-order columns show percentage-point changes. Traditional-task results average 25 task and prompt-style combinations. Node-classification results average six dataset-sampler groups. The last column counts comparisons with higher, equal, or lower accuracy than random order: 100 traditional-task comparisons or 24 node-classification comparisons per model.

| Task family | Model | Random | BFS | DFS | PR | PPR | Higher / equal / lower |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| Traditional | DeepSeek V4.1 Flash | 92.74 | +0.23 | −0.06 | +0.06 | +0.03 | 33 / 42 / 25 |
| Traditional | GLM-5.3 | 91.57 | −1.06 | −0.63 | −0.69 | −1.20 | 19 / 40 / 41 |
| Traditional | GPT-6 Luna | 82.54 | +1.17 | +0.51 | +1.97 | +2.49 | 54 / 17 / 29 |
| Node classification | DeepSeek V4.1 Flash | 68.00 | −0.22 | 0.00 | +2.00 | +2.44 | 15 / 2 / 7 |
| Node classification | GPT-6 Luna | 58.22 | −2.44 | −2.44 | −9.11 | −10.22 | 6 / 2 / 16 |

> **Finding 3.** Larger observed gains occur in specific settings, but none of the 300 traditional-task comparisons is significant after Holm correction.

Luna's weighted shortest-path results in Figure 4 illustrate this pattern: PPR increased accuracy under all five prompt styles, with gains of 2.86–13.57 percentage points. The few-shot comparison, which produced the largest of these gains, also had the smallest Holm-adjusted p-value across all 300 traditional-task comparisons. That value was 0.563, so neither this example nor any other comparison reached statistical significance after correction. The consistent positive direction across prompt styles was specific to Luna: for the same task and order, DeepSeek improved under four styles, while GLM declined under four and tied under CoT.

![PPR accuracy changes relative to random order on weighted shortest path, across five prompt styles and three models](</Users/taojiewang/Downloads/COMP5222 (L1) - Advanced Machine Learning with Graphs/figures/graphdo-audited-shortest-path-ppr.png>)

**Figure 4. PPR effects on weighted shortest path.** Bars show accuracy changes relative to random order (percentage points). Luna improved under all five prompt styles, DeepSeek under four, and GLM under none.

> **Finding 4.** An edge order can improve accuracy in one task family and reduce it in another, even for the same model.

Luna's results in Figure 5 illustrate this reversal: PPR increased mean accuracy by 2.49 percentage points on traditional graph tasks but reduced it by 10.22 points on node classification. The dataset and sampler breakdown in Table B1 of Appendix B shows that Luna's accuracy fell in five of the six settings. The sole improvement occurred on CiteSeer with Forest Fire sampling, where a 10.67-point gain outweighed the 8.00-point decline under ego sampling, leaving CiteSeer with a small positive dataset mean. Luna's predominantly negative node-classification results therefore differ from GraphDO's uniformly positive findings. [GraphDO Table 2](https://aclanthology.org/2025.acl-long.321.pdf#page=6) DeepSeek, by comparison, gained 2.44 points on node classification and changed by only +0.03 points on traditional tasks.

![PPR accuracy changes for DeepSeek and Luna, with family means and breakdowns by five traditional tasks and three citation datasets](</Users/taojiewang/Downloads/COMP5222 (L1) - Advanced Machine Learning with Graphs/figures/graphdo-audited-ppr-breakdown.png>)

**Figure 5. PPR has opposite effects across task families for Luna.** Bars show accuracy changes relative to random order (percentage points). Shaded rows show family means. Each task averages five prompt styles, and each dataset averages two sampling methods.

We hypothesize that an edge order helps when it emphasizes information that the model can use to solve the task. This explanation builds on GraphDO's attention-bias hypothesis and its experiment testing whether solution-like edge sequences can guide reasoning. [GraphDO §4.2](https://aclanthology.org/2025.acl-long.321.pdf#page=5) · [GraphDO §5](https://aclanthology.org/2025.acl-long.321.pdf#page=7) As Table 4 specifies, shortest-path PPR favors nodes on a computed minimum-weight path, whereas node-classification PPR uses distance from the target without label information. The shortest-path prior therefore identifies part of a solution in advance, while proximity to the target need not identify the most informative labeled nodes. If Luna relies on these prioritized nodes, PPR could aid path finding while drawing attention away from useful label evidence in node classification. DeepSeek's positive response on node classification suggests that models differ in how they use these cues, which could make PPR's emphasis on nearby nodes helpful for DeepSeek but misleading for Luna.

### 4.3 RQ2. Which tasks are most sensitive to order?

> **Finding 5.** Connectivity and cycle detection are consistently less sensitive to edge order than the three tasks requiring a vertex sequence.

The contrast between the two task groups holds across every model and prompt style. In all 15 model–prompt combinations, connectivity and cycle detection had lower variance across orders than weighted shortest path, Hamiltonian path, and topological sorting. Their mean variances in Table 8 were below 0.70 squared percentage points for all three models. This reproduces GraphDO's broad distinction between the two binary tasks and tasks requiring more extensive graph reasoning. We hypothesize that constructing a complete sequence exposes more opportunities to lose track of graph constraints as the edge order changes. For example, a Hamiltonian path must preserve adjacency while covering every vertex exactly once, whereas a positive connectivity answer can be established by finding any path between the queried vertices. [GraphDO §4.2 Q2](https://aclanthology.org/2025.acl-long.321.pdf#page=5)

**Table 8. Mean variance of accuracy across edge orders.** Each entry averages the population variances from the five prompt styles, in squared percentage points.

| Task | DeepSeek V4.1 Flash | GLM-5.3 | GPT-6 Luna |
| :--- | ---: | ---: | ---: |
| Connectivity | 0.07 | 0.07 | 0.67 |
| Cycle detection | 0.02 | 0.20 | 0.47 |
| Weighted shortest path | 2.29 | 5.20 | 15.76 |
| Hamiltonian path | 3.97 | 2.97 | 19.76 |
| Topological sorting | 0.70 | 3.87 | 14.52 |

> **Finding 6.** The most sensitive task depends on both the model and the prompt style.

Weighted shortest path does not retain its position as the most sensitive task across models. Averaged across prompt styles, Hamiltonian path had the highest variance for DeepSeek and Luna, at 3.97 and 19.76 squared percentage points, respectively. Only GLM retained GraphDO's shortest-path ranking, with a mean variance of 5.20. The prompt-level results further qualify these averages: shortest path ranked highest in only five of the 15 model–prompt combinations. Luna illustrates how the ranking changes within one model. Its highest variance occurred on shortest path under zero-shot prompting, Hamiltonian path under few-shot prompting, and topological sorting under CoT-BAG. Thus, task identity alone does not determine which problem is most vulnerable to reordering. [GraphDO Figure 4](https://aclanthology.org/2025.acl-long.321.pdf#page=6)

> **Finding 7.** Explicit reasoning does not consistently reduce sensitivity, and graph-construction instructions often increase it.

If order sensitivity partly reflects difficulty maintaining graph constraints, explicit reasoning might help models track those constraints more consistently. The prompt comparisons provide mixed support for this expectation. Adding the zero-shot CoT instruction lowered variance in six of the 15 model–task combinations, raised it in seven, and left two unchanged. Replacing answer-only few-shot examples with worked CoT examples lowered variance in nine combinations, although four increased and two were unchanged. Adding the graph-construction instruction to CoT was less favorable: CoT-BAG increased variance in 11 combinations and reduced it in only two. For Luna's topological sorting, variance rose from 11.31 under CoT to 30.82 under CoT-BAG. These results qualify GraphDO's observation that CoT does not remove order sensitivity: the response depends on the form of reasoning guidance, and explicitly reconstructing the graph does not ensure greater robustness to its presentation order. [GraphDO §4.2 Q2](https://aclanthology.org/2025.acl-long.321.pdf#page=6)

### 4.4 RQ3. Which orders work best for each task?

The highest-scoring order varied by task and model. On weighted shortest path, PPR led for DeepSeek and Luna, while random order led for GLM at 96.43%, as Table 9 shows. The original study ranked BFS first on that task. On topological sorting, PR led for DeepSeek and Luna at 99.86% and 90.00%, respectively, while DFS led for GLM at 87.43%. [GraphDO Table 1](https://aclanthology.org/2025.acl-long.321.pdf#page=5)

**Table 9. Traditional-task accuracy (%) averaged across five prompt styles.** Each entry summarizes 700 responses. Bold marks the highest accuracy for a model and task, including ties.

| Task | Model | Random | BFS | DFS | PR | PPR |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Connectivity | DeepSeek V4.1 Flash | **100.00** | **100.00** | **100.00** | 99.71 | 99.86 |
| Connectivity | GLM-5.3 | **100.00** | **100.00** | **100.00** | **100.00** | 99.71 |
| Connectivity | GPT-6 Luna | 98.71 | 99.14 | **99.29** | **99.29** | 98.14 |
| Cycle detection | DeepSeek V4.1 Flash | 99.86 | 99.86 | **100.00** | **100.00** | **100.00** |
| Cycle detection | GLM-5.3 | **99.86** | **99.86** | 99.14 | 99.71 | 99.71 |
| Cycle detection | GPT-6 Luna | 98.14 | **99.00** | 98.57 | 98.14 | **99.00** |
| Weighted shortest path | DeepSeek V4.1 Flash | 95.86 | 98.14 | 96.57 | 97.86 | **98.57** |
| Weighted shortest path | GLM-5.3 | **96.43** | 92.86 | 94.29 | 94.29 | 92.71 |
| Weighted shortest path | GPT-6 Luna | 78.71 | 80.71 | 79.00 | 83.14 | **88.00** |
| Hamiltonian path | DeepSeek V4.1 Flash | **69.14** | 68.14 | 67.86 | 66.57 | 65.71 |
| Hamiltonian path | GLM-5.3 | **76.00** | 75.57 | 73.86 | 74.29 | 72.86 |
| Hamiltonian path | GPT-6 Luna | 53.71 | **57.57** | 53.00 | 52.00 | 51.43 |
| Topological sorting | DeepSeek V4.1 Flash | 98.86 | 98.71 | 99.00 | **99.86** | 99.71 |
| Topological sorting | GLM-5.3 | 85.57 | 84.29 | **87.43** | 86.14 | 86.86 |
| Topological sorting | GPT-6 Luna | 83.43 | 82.14 | 85.43 | **90.00** | 88.57 |

The scores also varied when two orders produced identical prompts. For Hamiltonian path, a uniform PPR prior makes all 700 PR and PPR prompt pairs identical. DeepSeek's independently generated responses scored 66.57% under PR and 65.71% under PPR, with correctness differing in 66 of the 700 pairs. Since the edge sequences were identical, this difference reflects variation between responses rather than an effect of edge order. Such variation could therefore contribute to the small traditional-task differences. [Pinned ordering implementation](https://github.com/YuyaoGe/GraphDO/blob/4ba8f925c76574e2fca8d4f32f753b57a341b44f/graph_ordering.py)

On node classification, PR and PPR were the leading orders for DeepSeek, and random order and DFS led for Luna. PR ranked first in three dataset-sampler groups for DeepSeek, and PPR led or tied in the other three. Random order led in three groups for Luna, and DFS led or tied in the other three. Appendix B reports the group-level scores.

## 5. Discussion

Sensitivity to order does not imply that a structured order will improve accuracy. GLM illustrates the distinction: weighted shortest path was its most sensitive task, yet random order outperformed every structured order on that task. For DeepSeek and Luna, PPR instead ranked highest on shortest path. These results preserve GraphDO's broader finding that graph presentation matters unevenly across tasks, while showing that both the most sensitive task and the direction of an ordering effect depend on the model. The observed differences should also be read alongside the identical-prompt variation in §4.4, since a single response per traditional-task prompt does not separate generation variability from the effect of changing edge order. [GraphDO Table 1 and Figure 4](https://aclanthology.org/2025.acl-long.321.pdf#page=5)

Node classification gave a different picture. PPR improved DeepSeek's accuracy and lowered Luna's under the same dataset and sampler combinations. Both models had full response coverage on this task, so the contrast is observable without changes in the number of visible answers.

## 6. Limitations

The replication uses newly sampled graphs at half the original sample size and different hosted model interfaces. We use the public GraphDO prompt builder at commit `4ba8f925`. Some demonstration wording shown in the original paper differs from the public implementation. The original answers are unavailable in the public repository, which limits direct comparison of accuracy under a common scoring procedure. [GraphDO Table 4](https://aclanthology.org/2025.acl-long.321.pdf#page=16) · [Public implementation](https://github.com/YuyaoGe/GraphDO/tree/4ba8f925c76574e2fca8d4f32f753b57a341b44f)

Answer extraction remains a limit of the scoring procedure. On the traditional tasks, it extracted and checked 17,119 DeepSeek, 17,428 GLM, and 17,359 Luna answers out of 17,500 per model. On node classification, 1,572 of GLM's 2,250 responses were blank under its output cap. This makes that run unsuitable for comparison with the two models that returned answers for every prompt.

The five traditional tasks use small graphs and some labels are imbalanced. For example, 125 of the 140 cycle-detection graphs contain a cycle. For node classification, overlapping subgraphs and repeated prompts can reduce the number of independent observations. The 25 PubMed ego-subgraph draws, for example, contain 23 distinct graph and target pairs. [GraphDO graph generator](https://github.com/YuyaoGe/GraphDO/blob/4ba8f925c76574e2fca8d4f32f753b57a341b44f/graph_generator.py)

## 7. Conclusion

The half-scale replication recovers GraphDO's finding that edge-order sensitivity varies by task. Connectivity and cycle detection are consistently the least sensitive tasks, while the highest sensitivity occurs on Hamiltonian path for DeepSeek and Luna and weighted shortest path for GLM. The best edge order varies by task and model, and the observed gains over random order do not reach statistical significance after Holm correction across the 300 traditional-task comparisons. In node classification, PPR improves DeepSeek's accuracy and lowers Luna's, showing that an order that works well for one model can perform differently for another.

## References

Fagerland, M. W., Lydersen, S., & Laake, P. (2013). The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional. *BMC Medical Research Methodology, 13*, 91. [DOI](https://doi.org/10.1186/1471-2288-13-91)

Ge, Y., Liu, S., Bi, B., Wang, Y., Mei, L., Feng, W., Chen, L., & Cheng, X. (2025). *Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?* Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pp. 6404–6420. [ACL Anthology](https://aclanthology.org/2025.acl-long.321/)

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. [DOI](https://doi.org/10.2307/4615733)

## Appendix A. Reproducibility

The experiment is stored at `/mnt/new_ssd_3/taojie/comp5222-graphdo/half-scale` on the course project server. `experiment.json` records the design. The provenance files under `benchmark/traditional/` and `benchmark/node/` record the public commit, seed, package versions, and manifest hashes. Responses are keyed by case ID and model in the SQLite files under `runs/traditional/` and `runs/node/`. The 5,991 recovered GLM answers are stored under `repair-glm-4096/runs/traditional/`. The Forest Fire sampling compatibility fix is documented in `logs/node-sampler-compatibility.json`.

The report uses the case-level `scores` field in the local project file `analysis/audited_final_scores.json`. This snapshot supersedes `scratch/glm-final-report-20261122.json` after an audit of answer extraction. The audit uses the saved visible answers and graph metadata in `analysis/scoring_audit_inputs.json.gz`, without additional model calls. From the local project directory, run `python3 analysis/audit_scores.py`, followed by `python3 analysis/analyze_rq1.py` and `python3 analysis/analyze_rq2.py`. The scorer first reproduces all 52,500 previous scores, then applies the revised extraction rules. `analysis/scoring_audit.json` records every changed score, and `analysis/SCORING_AUDIT.md` documents the corrections and remaining fallback cases. Node-classification scores are unchanged. Figure generators in `figures/plot_*.py` read the audited results and analysis files.

## Appendix B. Node-Classification Accuracy by Dataset and Sampler

**Table B1. Node-classification accuracy (%).** Each cell contains 75 responses (25 sampled questions × three repeats). Bold marks each row's highest accuracy, including ties.

| Dataset | Sampler | Model | Random | BFS | DFS | PR | PPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora | Ego | DeepSeek V4.1 Flash | **64.00** | 60.00 | 58.67 | 58.67 | **64.00** |
| Cora | Ego | GPT-6 Luna | 52.00 | 50.67 | **57.33** | 52.00 | 45.33 |
| Cora | Forest Fire | DeepSeek V4.1 Flash | 76.00 | 77.33 | 73.33 | 78.67 | **82.67** |
| Cora | Forest Fire | GPT-6 Luna | **72.00** | 65.33 | 68.00 | 56.00 | 56.00 |
| CiteSeer | Ego | DeepSeek V4.1 Flash | 62.67 | 62.67 | 61.33 | **64.00** | 58.67 |
| CiteSeer | Ego | GPT-6 Luna | 44.00 | 48.00 | **58.67** | 38.67 | 36.00 |
| CiteSeer | Forest Fire | DeepSeek V4.1 Flash | 58.67 | 61.33 | 62.67 | **64.00** | 60.00 |
| CiteSeer | Forest Fire | GPT-6 Luna | 48.00 | **58.67** | **58.67** | 48.00 | **58.67** |
| PubMed | Ego | DeepSeek V4.1 Flash | 64.00 | 60.00 | 68.00 | 66.67 | **73.33** |
| PubMed | Ego | GPT-6 Luna | **56.00** | 49.33 | 37.33 | 45.33 | 42.67 |
| PubMed | Forest Fire | DeepSeek V4.1 Flash | 82.67 | 85.33 | 84.00 | **88.00** | 84.00 |
| PubMed | Forest Fire | GPT-6 Luna | **77.33** | 62.67 | 54.67 | 54.67 | 49.33 |
