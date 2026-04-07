# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: Most Underrated Novelty Benchmark (Custom)

### Overview
- **Source**: Created for this research project
- **Size**: 96 prompts (80 underrated + 16 control)
- **Format**: JSON and JSONL
- **Task**: Novelty evaluation of LLM responses
- **Categories**: movies, music, books, food, travel, technology, sports, science
- **License**: Research use

### Files
- `underrated_benchmark/prompts.json` — Full dataset with metadata
- `underrated_benchmark/prompts.jsonl` — Line-delimited format for easy iteration

### Loading the Dataset

```python
import json

# Load full dataset
with open("datasets/underrated_benchmark/prompts.json") as f:
    dataset = json.load(f)

# Iterate over prompts
for category, prompts in dataset["underrated_prompts"].items():
    for prompt in prompts:
        print(f"[{category}] {prompt}")

# Or load JSONL for flat iteration
with open("datasets/underrated_benchmark/prompts.jsonl") as f:
    for line in f:
        item = json.loads(line)
        print(f"[{item['type']}:{item['category']}] {item['prompt']}")
```

### Dataset Design

The dataset is structured to test the hypothesis that LLMs converge on popular "underrated" answers:

1. **Underrated prompts** (80): "What is the most underrated X?" across 8 categories, 10 prompts each
2. **Control prompts** (16): Split into:
   - "popular_consensus" (8): "What is the best X?" — expected high convergence
   - "factual_baseline" (8): "What is the most popular/highest-grossing X?" — expected near-perfect convergence

By comparing response diversity across these three prompt types, we can measure whether "underrated" prompts produce more or less diversity than "best" and factual questions.

### Experimental Protocol

For each prompt:
1. Query multiple LLMs (≥5 different model families)
2. Generate ≥20 responses per model per prompt (varying temperature)
3. Extract the named item from each response
4. Measure convergence rate, semantic diversity, and popularity correlation

### Notes
- Prompts are designed to elicit single-item answers for easy comparison
- Categories span subjective (movies) to semi-objective (science) domains
- Control prompts provide calibration for expected convergence levels
