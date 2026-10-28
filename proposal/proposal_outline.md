# Proposal Outline

## Working Title

**Revisiting Graph Reasoning in Modern Large Language Models: Reproduction and Extension of GraphQA**

## 1. Abstract

- Problem: the reliability of LLM reasoning over graph-structured information.
- Prior work: *Talk Like a Graph* and the GraphQA benchmark.
- Proposed study: reproduce selected findings with modern LLMs and extend the evaluation to complexity, representation robustness, and inference cost.
- Main methods, expected findings, and practical significance.

## 2. Introduction

### 2.1 Motivation

- Importance of graph reasoning in structured data applications.
- Difference between natural-language reasoning and explicit relational reasoning.
- Why the progress of modern LLMs motivates a new evaluation.

### 2.2 Research Gap

- Existing GraphQA results primarily reflect earlier-generation LLMs.
- Limited analysis of representation invariance and token efficiency.
- Need for a controlled, reproducible comparison across modern models.

### 2.3 Contributions

- Reproduction of representative GraphQA experiments.
- Controlled comparison of graph encodings and graph tasks.
- Analysis of graph complexity and representation robustness.
- Accuracy–cost analysis based on token usage and API measurements.
- Release of reproducible experiment code and evaluation artifacts.

## 3. Background and Related Work

### 3.1 LLM Reasoning over Structured Data

### 3.2 Graph-to-Text Encoding

### 3.3 GraphQA and *Talk Like a Graph*

- Benchmark tasks.
- Graph generators.
- Encoding strategies.
- Main findings of the original study.

### 3.4 Evaluation of Modern LLMs

- Model comparison and API-based evaluation.
- Reproducibility and cost considerations.

## 4. Research Questions

### RQ1: Model Capability

Have modern LLMs improved their performance on basic graph reasoning tasks compared with the historical GraphQA results?

### RQ2: Encoding Sensitivity

Does the choice of graph-to-text encoding continue to affect reasoning accuracy in modern LLMs?

### RQ3: Graph Complexity

How do graph size, density, and topology affect the performance of modern LLMs?

### RQ4: Representation Robustness

Are model predictions invariant to semantically equivalent changes in node names, edge order, and neighbor-list order?

### RQ5: Accuracy–Cost Trade-off

How do different graph encodings trade off reasoning accuracy, input/output token usage, latency, and monetary cost?

## 5. Benchmark and Task Definition

### 5.1 Graph Generation

- Graph generators: ER, BA, SFN, SBM, path, star, and complete graphs.
- Node-count and density ranges.
- Directed versus undirected graph policy.
- Random seeds and split construction.

### 5.2 GraphQA Tasks

- Edge existence.
- Node degree.
- Cycle detection.
- Optional extension tasks, if feasible.

### 5.3 Ground-Truth Computation

- Deterministic NetworkX algorithms.
- Answer normalization and exact evaluation rules.
- Balanced sampling for binary tasks.

## 6. Experimental Methodology

### 6.1 LLM Selection

- Modern API-accessible models.
- Model identifiers, versions, and configuration.
- Rationale for model selection.

### 6.2 Graph Encodings

- Adjacency encoding.
- Incident encoding.
- Natural-language relation encodings.
- Encoding-specific prompt templates.

### 6.3 Prompting Protocol

- Fixed system and user prompt structure.
- Output-format constraints.
- Temperature and decoding settings.
- Retry and failure-handling policy.

### 6.4 Experimental Parts

#### Part I: Reproduction

- Reproduce representative encoding and task comparisons.
- Compare modern results with historical findings.

#### Part II: Structural Analysis

- Vary graph size, density, and topology.
- Analyze task-specific difficulty.

#### Part III: Representation Robustness

- Apply semantics-preserving permutations and renamings.
- Measure accuracy and prediction consistency.

#### Part IV: Efficiency Analysis

- Record input tokens, output tokens, latency, and cost.
- Compare accuracy–efficiency trade-offs.

## 7. Evaluation and Statistical Analysis

### 7.1 Primary Metrics

- Accuracy.
- Per-task and per-topology accuracy.
- Prediction consistency under equivalent representations.

### 7.2 Efficiency Metrics

- Input token count.
- Output token count.
- Latency.
- Estimated API cost.

### 7.3 Baselines and Controls

- Majority baseline for imbalanced tasks.
- Paired evaluation on identical graph instances.
- Fixed random seeds.
- Multiple runs where API nondeterminism requires them.

### 7.4 Statistical Reporting

- Confidence intervals.
- Paired model/encoding comparisons.
- Error analysis and representative failure cases.

## 8. Reproducibility and Implementation Plan

- Repository structure.
- Graph generation and encoding modules.
- API query runner.
- Result schema and experiment manifest.
- Configuration and secret-management policy.
- Reproduction commands and environment specification.

## 9. Expected Results and Discussion Plan

- Whether modern models outperform historical baselines.
- Whether encoding effects remain substantial.
- Which graph structures are most challenging.
- Whether equivalent textual representations produce inconsistent answers.
- Whether higher accuracy requires greater token or monetary cost.

## 10. Limitations and Ethical Considerations

- Differences between historical and current model families.
- API version drift and incomplete billing information.
- Synthetic graphs versus real-world graphs.
- Limited graph sizes and task coverage.
- Privacy and secret-handling requirements for API experiments.

## 11. Project Schedule

- Benchmark and environment setup.
- Pilot experiments.
- Reproduction experiments.
- Extension experiments.
- Analysis and report writing.
- Final code and presentation.

## 12. Deliverables

- Reproducible codebase.
- Experiment configurations and manifests.
- Raw and processed evaluation results.
- Final report.
- Presentation slides.

## 13. References

- *Talk Like a Graph: Encoding Graphs for Large Language Models*.
- GraphQA and related graph-reasoning literature.
- Documentation for the evaluated LLM APIs.
