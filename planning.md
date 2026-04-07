# Research Plan: 'Most Underrated' as a Novelty Metric

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used for creative and generative tasks, yet their ability to produce genuinely novel outputs is poorly understood. If LLMs systematically converge on the same "underrated" items—items that are already widely cited as underrated (e.g., on Reddit)—this reveals a fundamental limitation in their capacity for taste, discernment, and innovation. Understanding this gap is critical for anyone relying on LLMs for ideation, recommendation, or creative work.

### Gap in Existing Work
The literature extensively documents LLM output homogeneity (generative monoculture, creative homogeneity across models) and popularity bias in recommendations. However:
1. **No benchmark tests "underrated" convergence** — no dataset specifically asks LLMs for underrated items and measures whether answers are truly novel vs. commonly cited.
2. **The "meta-popularity" problem is unexplored** — items frequently called "underrated" are paradoxically popular-as-underrated, creating a second-order popularity bias.
3. **No cross-model comparison on subjective preference questions** — prior work focuses on creativity tasks (AUT, DAT) or factual QA, not opinion/taste questions.

### Our Novel Contribution
We introduce the "most underrated" prompt as a diagnostic for LLM novelty. We:
1. Create a benchmark of 80 "underrated" prompts across 8 categories + 16 controls
2. Query multiple LLMs (GPT-4.1, GPT-4.1-mini, GPT-4o) with N=20 samples per prompt
3. Measure inter- and intra-model convergence using semantic embeddings
4. Compare convergence on "underrated" vs. "best" vs. factual prompts
5. Scrape Reddit to build a "commonly cited as underrated" reference set for ground-truth comparison

### Experiment Justification
- **Experiment 1 (Multi-model querying)**: Tests whether different LLMs converge on the same "underrated" answers, extending monoculture findings to subjective taste questions.
- **Experiment 2 (Intra-model diversity)**: Tests whether a single LLM produces diverse answers across runs, measuring the depth of its "taste."
- **Experiment 3 (Underrated vs. Control comparison)**: Tests whether "underrated" prompts elicit less diversity than "best" or factual prompts—if so, the novelty gap is specific to taste/discernment.
- **Experiment 4 (Reddit baseline comparison)**: Tests whether LLM answers match commonly-cited Reddit answers, directly measuring whether LLM "underrated" = internet consensus "underrated."

## Research Question
Do LLMs converge on the same "obvious" underrated answers when asked to name the most underrated item in a category, and does this convergence exceed what we see for other subjective or factual questions?

## Hypothesis Decomposition
- H1: LLMs show high inter-model convergence on "underrated" prompts (different models give similar answers)
- H2: LLMs show low intra-model diversity on "underrated" prompts (same model gives same answer repeatedly)
- H3: Convergence on "underrated" prompts is comparable to or exceeds convergence on "best" prompts
- H4: LLM "underrated" answers correlate with commonly-cited Reddit answers

## Proposed Methodology

### Approach
Query multiple OpenAI models (GPT-4.1, GPT-4.1-mini, GPT-4o) with each of the 96 prompts, collecting 20 responses per prompt per model at temperature=1.0. Use sentence embeddings (all-MiniLM-L6-v2) to measure semantic diversity. Compare against Reddit-sourced baseline answers.

### Experimental Steps
1. Query 3 models × 96 prompts × 20 samples = 5,760 API calls
2. Extract the specific item named from each response
3. Compute intra-model diversity (pairwise cosine similarity within model)
4. Compute inter-model diversity (pairwise cosine similarity across models)
5. Compare "underrated" vs "best" vs "factual" prompt types
6. Scrape/collect Reddit "most underrated" threads for baseline
7. Measure overlap between LLM answers and Reddit answers

### Baselines
- "Best X" control prompts (consensus questions)
- Factual control prompts (questions with objective answers)
- Cross-model comparison (each model as baseline for others)

### Evaluation Metrics
1. **Unique answer rate**: Proportion of distinct answers per prompt per model
2. **Semantic diversity**: Mean pairwise cosine distance of sentence embeddings
3. **Shannon entropy**: Over extracted answer distribution
4. **Inter-model overlap**: Jaccard similarity of answer sets across models
5. **Reddit overlap**: Fraction of LLM answers appearing in Reddit threads

### Statistical Analysis Plan
- Welch's t-test comparing diversity metrics between prompt types
- Effect sizes (Cohen's d) for all comparisons
- Bootstrap confidence intervals for diversity metrics
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- LLMs will show high convergence on "underrated" prompts (~3-5 unique answers per prompt across 20 samples)
- Convergence will be comparable to "best" prompts and higher than factual prompts
- Inter-model overlap will be high (>50% of top answers shared)
- Strong overlap with Reddit commonly-cited answers

## Timeline
- Phase 1 (Planning): 15 min ✓
- Phase 2 (Setup): 10 min
- Phase 3 (Implementation): 60 min
- Phase 4 (Experiments): 60 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- API rate limits → use exponential backoff
- Response parsing → structured prompts requesting single item
- Reddit data quality → manual curation of top answers
- Cost → ~5,760 calls at ~$0.002/call ≈ $12

## Success Criteria
- Successfully query ≥3 models on all 96 prompts with ≥20 samples each
- Measure statistically significant convergence patterns
- Produce clear visualizations showing convergence differences across prompt types
- Quantify overlap with Reddit baseline
