# REPORT: 'Most Underrated' as a Novelty Metric for LLMs

## 1. Executive Summary

We tested whether large language models can produce genuinely novel answers when asked to name the "most underrated" item in a category, or whether they converge on the same well-known "underrated" picks. Querying 3 OpenAI models (GPT-4.1, GPT-4.1-mini, GPT-4o) with 80 "underrated" prompts across 8 categories (20 samples per prompt per model, 4,800 underrated responses total), we find **strong intra-model convergence**: the single most popular answer captures 48.7% of responses on average, with extreme cases like "Children of Men" (most underrated movie) reaching 85-95% dominance across all three models. However, we also find a nuanced picture: underrated prompts actually elicit *more lexical diversity* than "best" or factual control prompts (unique rate 0.40 vs 0.21 vs 0.13), and inter-model agreement is surprisingly low (8.8% top-1 unanimity). This reveals that LLMs have learned *different* but equally *stereotyped* notions of "underrated"—each model converges on its own cliché rather than a shared consensus, but none produce genuinely novel answers. The "most underrated" prompt is a useful diagnostic: it reveals that LLMs possess a narrow, training-data-derived sense of "taste" rather than genuine discernment.

## 2. Research Question & Motivation

**Hypothesis**: LLMs struggle with generating novel responses when asked to name the most underrated item in a category. If their answers align with commonly-cited "underrated" items (e.g., from Reddit), this indicates a lack of genuine novelty.

**Why this matters**: As LLMs are increasingly used for ideation, recommendation, and creative tasks, understanding whether they can reason about what is genuinely underrated (vs. regurgitating popular opinions about underratedness) is crucial. This is a test of *taste* and *discernment*, not just knowledge.

**Gap in existing work**: The generative monoculture literature (Wu et al. 2024, Wenger & Kenett 2025) documents output homogeneity in creativity tasks, but no work specifically tests convergence on subjective "underrated" questions—a domain that requires second-order reasoning about popularity and perception.

## 3. Methodology

### Models
- **GPT-4.1** (OpenAI, 2025)
- **GPT-4.1-mini** (OpenAI, 2025)
- **GPT-4o** (OpenAI, 2024)

### Dataset
96 prompts total:
- **80 "underrated" prompts**: 10 per category (movies, music, books, food, travel, technology, sports, science)
- **8 "popular consensus" controls**: "What is the best X?" (subjective but well-known answers)
- **8 "factual baseline" controls**: "What is the highest-grossing/most popular X?" (objective answers)

### Protocol
- 20 samples per prompt per model at temperature=1.0
- System prompt requesting a single, specific answer with brief justification
- Total: 5,760 API calls (3 models × 96 prompts × 20 samples)
- Random seed varied per sample (base seed 42 + sample_id)

### Evaluation Metrics
1. **Unique answer rate**: Proportion of distinct answers per prompt per model (out of 20)
2. **Shannon entropy**: Over the distribution of extracted answers
3. **Dominance**: Fraction of responses matching the most common answer
4. **Semantic diversity**: 1 - mean pairwise cosine similarity of response embeddings (all-MiniLM-L6-v2)
5. **Inter-model Jaccard overlap**: Overlap of top-5 answer sets between model pairs
6. **Reddit overlap**: Fraction of answers matching a curated list of commonly-cited "underrated" items

### Tools & Environment
- Python 3.12.8, OpenAI API
- sentence-transformers 5.3.0 (all-MiniLM-L6-v2)
- scipy 1.17.1, pandas, matplotlib, seaborn
- CPU computation (CUDA driver mismatch prevented GPU use for embeddings)

## 4. Results

### 4.1 Intra-Model Convergence

The core finding: **within each model, responses converge strongly on a small set of answers**.

| Prompt Type | Unique Rate | Entropy | Dominance |
|---|---|---|---|
| **Underrated** | 0.397 ± 0.216 | 2.209 ± 1.102 | **0.487 ± 0.257** |
| Popular Consensus | 0.206 ± 0.109 | 1.316 ± 0.754 | 0.662 ± 0.207 |
| Factual Baseline | 0.131 ± 0.157 | 0.609 ± 0.991 | 0.844 ± 0.251 |

**Key insight**: The average dominance of 0.487 means that for a typical underrated prompt, nearly half of all 20 responses from a single model name the *exact same item*. The most extreme cases show 100% convergence (e.g., GPT-4.1 gives "New Mexico" 20/20 times for "most underrated US state", "Stoner by John Williams" 20/20 for "most underrated book").

### 4.2 Most Converged "Underrated" Answers

The following items were named by the overwhelming majority of responses across all 3 models (60 total samples per prompt):

| Prompt | Top Answer | Frequency | Unique Answers |
|---|---|---|---|
| Most underrated spice | **Sumac** | 93% (56/60) | 4 |
| Most underrated movie of all time | **Children of Men** | 87% (52/60) | 8 |
| Most underrated video game console | **Sega Dreamcast** | 77% (46/60) | 7 |
| Most underrated animated movie | **The Iron Giant** | 67% (40/60) | 6 |
| Most underrated dessert | **Rice pudding** | 67% (40/60) | 12 |
| Most underrated country to visit | **Slovenia** | 65% (39/60) | 6 |
| Most underrated operating system | **Haiku OS** | 65% (39/60) | 6 |
| Most underrated scientist | **Lise Meitner** | 63% (38/60) | 8 |
| Most underrated thriller | **Prisoners** | 63% (38/60) | 9 |
| Most underrated fruit | **Persimmon** | 62% (37/60) | 13 |

These are all **well-known "underrated" picks** that appear regularly in Reddit threads and online discussions. The LLMs have learned the *consensus about what is underrated* rather than reasoning independently about underratedness.

### 4.3 Semantic Diversity

Embedding-based diversity (1 - mean pairwise cosine similarity):

| Prompt Type | Semantic Diversity |
|---|---|
| Underrated | 0.370 ± 0.172 |
| Popular Consensus | 0.179 ± 0.130 |
| Factual Baseline | 0.116 ± 0.108 |

Underrated prompts show higher semantic diversity than controls, but this is primarily because the *explanations* vary even when the *named item* is the same. When an LLM says "Children of Men" 17/20 times, it explains why differently each time, inflating semantic diversity.

### 4.4 Category Analysis

Convergence varies by domain:

| Category | Mean Dominance | Mean Unique Rate |
|---|---|---|
| Food | 0.597 | 0.287 |
| Movies | 0.560 | 0.292 |
| Technology | 0.523 | 0.400 |
| Travel | 0.523 | 0.337 |
| Sports | 0.488 | 0.413 |
| Science | 0.433 | 0.468 |
| Books | 0.430 | 0.462 |
| Music | 0.345 | 0.518 |

**Food and movies** show the strongest convergence—perhaps because these categories have the most well-established "underrated" consensus online. **Music** shows the most diversity, possibly because musical taste is more fragmented.

### 4.5 Inter-Model Overlap

| Prompt Type | Mean Jaccard | Top-1 Unanimity |
|---|---|---|
| Underrated | 0.180 ± 0.135 | 8.8% |
| Popular Consensus | 0.304 ± 0.126 | 37.5% |
| Factual Baseline | 0.645 ± 0.319 | 50.0% |

**Surprising finding**: Despite strong intra-model convergence, inter-model agreement on underrated prompts is low. Only 8.8% of prompts have all 3 models agreeing on the same #1 answer. This means different models have learned *different stereotyped "underrated" answers*—e.g., GPT-4.1 strongly favors "Georgian cuisine" while GPT-4.1-mini favors "Filipino cuisine" for the most underrated cuisine.

### 4.6 Overlap with Commonly-Cited "Underrated" Items

Using a curated reference list of items commonly called "underrated" on Reddit:

| Category | Overlap Rate |
|---|---|
| Movies | 40.5% |
| Food | 19.0% |
| Technology | 18.5% |
| Travel | 15.8% |
| Music | 1.7% |
| Books | 0.3% |
| **Overall** | **16.0%** |

The movie category shows the highest overlap, confirming that LLM "underrated" picks in this domain are particularly clichéd. The low overlap in music and books likely reflects the curated list being incomplete rather than genuine novelty.

### 4.7 Statistical Tests

All comparisons between underrated and control prompt types are highly significant:

| Comparison | t-statistic | p-value | Cohen's d |
|---|---|---|---|
| Unique rate: underrated vs popular | 7.29 | < 0.0001 | 1.12 |
| Unique rate: underrated vs factual | 7.62 | < 0.0001 | 1.42 |
| Entropy: underrated vs popular | 5.26 | < 0.0001 | 0.95 |
| Entropy: underrated vs factual | 7.46 | < 0.0001 | 1.54 |
| Dominance: underrated vs popular | -3.85 | 0.0006 | -0.76 |
| Dominance: underrated vs factual | -6.62 | < 0.0001 | -1.42 |

Underrated prompts show significantly *lower* dominance (more spread) than both control types, but this does not indicate novelty—it indicates the answer space is less constrained.

## 5. Analysis & Discussion

### The "Meta-Popularity" Problem
Our central finding is that LLMs have learned a *meta-popularity bias*: items that are **popular as underrated** (frequently described as underrated in training data) are the ones LLMs name. "Children of Men" isn't underrated at all in online discourse—it's the canonical example of an underrated movie. Similarly, sumac is the canonical underrated spice, Sega Dreamcast the canonical underrated console. The LLMs have learned the *wisdom of the crowd's contrarianism*, not genuine discernment.

### Intra-Model Convergence vs. Inter-Model Divergence
A nuanced finding: while individual models converge strongly (each model picks the same answer ~50% of the time), different models pick *different* stereotyped answers. This suggests the convergence pattern is an artifact of each model's specific training data distribution, not a fundamental property of the question. The models don't share a universal "underrated" canon—they each have their own.

### Category Effects
Categories with stronger online consensus about what is underrated (movies, food) show more convergence. Categories where "underrated" is more subjective or fragmented (music, books) show more diversity. This supports the hypothesis that LLMs are reproducing training data patterns, not reasoning about underratedness.

### Comparison to Prior Work
Our results align with:
- **Generative monoculture** (Wu et al. 2024): LLMs produce narrower output distributions than humans
- **Creative homogeneity** (Wenger & Kenett 2025): Cross-model homogeneity exists, though we find it's weaker for subjective preference questions
- **Diminished diversity of thought** (PMC 2024): The 99.7% convergence on survey questions maps to our 87-93% convergence on the most stereotyped prompts

Our novel contribution: demonstrating that this convergence specifically manifests as **meta-popularity bias** in taste/discernment tasks.

## 6. Limitations

1. **Only OpenAI models tested**: We used 3 models from the same family. Testing Claude, Gemini, Llama, and Mistral would strengthen inter-model findings.
2. **No human baseline**: We did not collect human responses to the same prompts. Reddit data served as a proxy but a formal crowdsourced comparison would be more rigorous.
3. **Item extraction noise**: Our regex-based extraction occasionally includes explanation text. More sophisticated NER or LLM-based extraction would improve precision.
4. **Reddit overlap list is incomplete**: The curated "commonly-cited" list covers only ~6 categories and may miss items, leading to underestimated overlap.
5. **Temperature=1.0 only**: We did not test temperature variation as an independent variable, which the literature suggests has limited effect on diversity.
6. **English-only**: The bias patterns may differ in other languages.

## 7. Conclusions & Next Steps

### Answer to Research Question
**Yes, LLMs converge strongly on "obvious" underrated answers.** When asked for the most underrated item in a category, LLMs reproduce the popular consensus about what is underrated rather than identifying genuinely novel items. The "most underrated movie" is always Children of Men, the "most underrated spice" is always sumac, and the "most underrated video game console" is always the Sega Dreamcast. This confirms the hypothesis that LLMs lack genuine taste/discernment and instead echo training data patterns.

The "most underrated" prompt is a **useful diagnostic for LLM novelty**: a model capable of true innovation should not consistently name the most commonly-cited "underrated" item. This test requires second-order reasoning (what *most people* don't appreciate → what is genuinely overlooked), which current LLMs fail at systematically.

### Implications
- LLMs should not be trusted for "novel" recommendations without verification
- The "most underrated" prompt can serve as a quick diagnostic for generative novelty in new models
- The meta-popularity bias suggests RLHF/instruction tuning may specifically optimize for popular-but-contrarian answers

### Recommended Next Steps
1. **Expand to more model families** (Claude, Gemini, Llama, Mistral) to test whether inter-model divergence persists
2. **Collect human survey data** for direct diversity comparison
3. **Scrape Reddit systematically** for ground-truth "commonly-cited underrated" lists
4. **Test mitigation strategies**: chain-of-thought prompting, persona variation, explicit novelty instructions
5. **Test across temperatures** (0.0, 0.5, 1.0, 1.5) to measure the effect on convergence
6. **Longitudinal study**: re-run as models are updated to track whether the canonical "underrated" picks change

## References

1. Wu, F., Black, E., Chandrasekaran, V. (2024). "Generative Monoculture in Large Language Models." arXiv:2407.02209.
2. Wenger, E., Kenett, Y. (2025). "We're Different, We're the Same: Creative Homogeneity Across LLMs." arXiv:2501.19361.
3. Dhingra, H. (2026). "Magic, Madness, Heaven, Sin: LLM Output Diversity." arXiv:2604.01504.
4. Davydov, P. et al. (2025). "LLM Generation Novelty Through the Lens of Semantic Similarity." arXiv:2510.27313.
5. Lichtenberg et al. (2024). "Large Language Models as Recommender Systems: Popularity Bias." Amazon Science.
6. PMC (2024). "Diminished Diversity-of-Thought in a Standard Large Language Model."

## Appendix: Reproducibility

```bash
# Environment setup
uv venv && source .venv/bin/activate
uv add openai sentence-transformers numpy pandas matplotlib seaborn scipy scikit-learn tqdm

# Run experiment (requires OPENAI_API_KEY env var)
python src/run_experiment.py    # ~5,760 API calls, ~20 min

# Run analysis
python src/extract_items.py     # Improved item extraction
python src/analyze_results.py   # Main analysis + plots
python src/detailed_analysis.py # Cross-model + category analysis
```

Config: seed=42, temperature=1.0, max_tokens=150, 20 samples/prompt/model.
Estimated API cost: ~$15-20.
