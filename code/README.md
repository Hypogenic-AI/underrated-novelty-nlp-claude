# Cloned Repositories

## Repo 1: GeMo (Generative Monoculture)
- **URL**: https://github.com/GeMoLLM/GeMo
- **Purpose**: Official implementation of "Generative Monoculture in Large Language Models" paper
- **Location**: code/GeMo/
- **Key files**: Contains code for measuring diversity metrics (sentiment, topic, algorithms) between source and generated distributions
- **Notes**: Provides reusable framework for measuring dispersion metrics. Key methodology: attribute extraction + dispersion comparison between D_src and D_gen.

## Repo 2: Divergent Thinking Benchmark
- **URL**: https://github.com/lechmazur/divergent
- **Purpose**: LLM Divergent Thinking Creativity Benchmark — LLMs generate 25 unique words starting with a given letter
- **Location**: code/divergent/
- **Key files**: Contains benchmark results for many LLMs, providing a reference for divergent thinking scores
- **Notes**: Useful as comparison data. Shows how different LLMs score on divergent thinking tasks. Contains extensive model comparison data.

## Repo 3: LiveIdeaBench
- **URL**: https://github.com/x66ccff/liveideabench
- **Purpose**: Evaluating LLMs' Scientific Creativity and Idea Generation with Minimal Context (Nature Communications)
- **Location**: code/liveideabench/
- **Key files**: Benchmark with 1,180 keywords across 22 scientific domains, evaluation scripts
- **Notes**: Published in Nature Communications. Assesses originality, feasibility, fluency, flexibility, and clarity. Can provide methodology for evaluating novelty dimensions.

## Relevance to Our Research

These repositories provide:
1. **Metrics infrastructure** (GeMo): Dispersion metrics, attribute extraction, comparison framework
2. **Baseline data** (divergent): LLM divergent thinking scores across many models
3. **Evaluation methodology** (LiveIdeaBench): Multi-dimensional novelty assessment framework

For our "most underrated" experiment, we can adapt:
- GeMo's dispersion measurement approach to compare LLM responses to human survey data
- The divergent benchmark's multi-model evaluation pattern
- LiveIdeaBench's scoring dimensions (especially originality and flexibility)
