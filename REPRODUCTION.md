# Talk Like a Graph：本地复现说明

`talk-like-a-graph/` 是论文 *Talk Like a Graph: Encoding Graphs for Large
Language Models* 的开源代码。当前仓库可以直接复现图生成、图文本编码和若干图任务的答案生成；论文中使用 PaLM 62B/PaLM 2 的完整准确率表不能在本地原样重跑，因为模型服务和部分数据生成脚本并未随仓库提供。

## 可以复现的内容

1. 生成不同结构的图：Erdős–Rényi（`er`）、Barabási–Albert（`ba`）、随机块模型（`sbm`）、scale-free network（`sfn`）、完全图（`complete`）、星形图（`star`）和路径图（`path`）。
2. 将同一张图编码成不同文本形式：`adjacency`、`incident`、`friendship`、`coauthorship`、`social_network`、`expert` 等。
3. 生成基础图任务及其确定答案：边存在性、节点度数、节点/边计数、连通节点、不连通节点、环检测、可达性、最短路径、三角形计数和最大流等。
4. 将生成的图问题文本交给本地 LLM API，再统计回答准确率。项目根目录的 `query_llms.py` 提供了 GLM、DeepSeek 和 Codex-compatible 接口的统一调用方式。

## 安装轻量依赖

仓库源码使用 Python 3.10+ 的类型标注。图生成和编码部分只需要 Python、NetworkX、NumPy 和 `absl-py`：

```bash
cd talk-like-a-graph
python3.11 -m venv .venv
.venv/bin/pip install networkx numpy absl-py
```

完整的任务文件生成器还会导入 TensorFlow、TensorFlow-GNN 和 SeqIO；如果只做下面的图生成、编码和任务答案示例，不需要安装这几个大型依赖。

## 生成 GraphML 图文件

```bash
PYTHONPATH=. .venv/bin/python -m talk_like_a_graph.graph_generators_runner \
  --algorithm path \
  --number_of_graphs 10 \
  --output_path /tmp/tlag-run \
  --split validation
```

输出目录类似：

```text
/tmp/tlag-run/undirected/path/validation/0.graphml
```

`--split` 可用 `train`、`validation`、`test`；不同 split 使用固定随机种子。`--directed` 可以生成有向图。

## 生成图文本和任务答案

```python
from talk_like_a_graph import graph_generators, graph_tasks, graph_text_encoders

graph = graph_generators.generate_graphs(
    number_of_graphs=1,
    algorithm="path",
    directed=False,
    random_seed=123,
)[0]

text = graph_text_encoders.encode_graph(graph, graph_encoder="incident")
example = graph_tasks.NodeCount().prepare_examples_dict(
    [graph], ["path"], "incident"
)[0]

print(text)
print(example["question"])
print(example["answer"])
```

这一步可以验证编码方法和图任务的逻辑，不需要调用 LLM。`incident` 编码会逐节点列出相邻节点，适合复现论文中节点度数和连通节点任务的输入形式。

## 接入本地 LLM

先在项目根目录查看可用配置：

```bash
python3 query_llms.py --list
```

然后可以把上一步生成的 `example["question"]` 保存或直接传给：

```bash
python3 query_llms.py --profile claude-glm --prompt "在这里放入图问题文本"
python3 query_llms.py --profile claude-ds --prompt "在这里放入图问题文本"
python3 query_llms.py --profile codex --prompt "在这里放入图问题文本"
```

要复现准确率，需要对一批随机图重复生成问题、记录模型回答，并根据任务类型解析答案。论文中的 GraphQA 评测主要比较的就是编码方式、提问方式、图结构和模型规模对准确率的影响。

本项目还提供了一个 10 图环检测小实验：

```bash
PYTHONPATH=. talk-like-a-graph/.venv311/bin/python \
  experiments/graph_api_trial.py
```

它使用固定随机种子和 `incident` 编码，分别询问 GLM、DeepSeek 和 Codex
接口，并将原始回答保存到 `experiments/graph_api_trial_results.json`。Codex
网关通过其 OpenAI-compatible `/v1/chat/completions` 路由调用。

## 当前限制

- 论文实验依赖 PaLM 62B/PaLM 2 服务；本地仓库不包含这些模型权重或原始服务配置。
- README 中提到的 `graphqa/graph_generator.sh` 和 `task_generator.sh` 不在当前公开仓库中，需要使用 Python 模块入口。
- 仓库自带的两个测试文件使用了旧式 `absl` 测试类继承方式，在现代 `absl-py` 下会触发 MRO 错误；图生成、编码和任务逻辑本身可以独立运行。
- `graph_generators_runner.py` 原本使用了不存在的 `os.Open`；本仓库已修正为带上下文管理的内置 `open`。
