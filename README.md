# 'Most Underrated' as a Novelty Metric for LLMs

Testing whether LLMs can identify genuinely underrated items or just echo the popular consensus about what is underrated.

## Key Findings

- **Strong convergence**: LLMs name the same "underrated" item ~49% of the time on average, with extremes like "Children of Men" (87%), "sumac" (93%), and "Sega Dreamcast" (77%)
- **Meta-popularity bias**: The items LLMs call "underrated" are the most commonly-cited "underrated" picks online — they've learned the crowd's contrarianism, not genuine discernment
- **Model-specific stereotypes**: Different models converge on different items (8.8% cross-model agreement), suggesting training data artifacts rather than objective reasoning
- **Category variation**: Food and movies show highest convergence (~56-60% dominance); music shows lowest (~35%)
- **The "most underrated" prompt is a useful novelty diagnostic**: any model that consistently names Children of Men as the most underrated movie is reproducing, not innovating

## Data

- 3 models (GPT-4.1, GPT-4.1-mini, GPT-4o) x 96 prompts x 20 samples = 5,760 responses
- 80 "underrated" prompts across 8 categories + 16 control prompts
- Full results in `results/`, visualizations in `figures/`

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv add openai sentence-transformers numpy pandas matplotlib seaborn scipy scikit-learn tqdm
export OPENAI_API_KEY=your_key
python src/run_experiment.py
python src/extract_items.py
python src/analyze_results.py
python src/detailed_analysis.py
```

## File Structure

```
├── REPORT.md                    # Full research report
├── planning.md                  # Research plan
├── literature_review.md         # Literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── run_experiment.py        # Query LLMs (5,760 API calls)
│   ├── extract_items.py         # Clean item name extraction
│   ├── analyze_results.py       # Main analysis + stats + plots
│   └── detailed_analysis.py     # Cross-model + category analysis
├── datasets/underrated_benchmark/
│   └── prompts.json             # 96-prompt benchmark
├── results/                     # Raw + processed results, CSVs
├── figures/                     # Visualizations
├── papers/                      # Reference papers (PDFs)
└── code/                        # Reference codebases
```

See [REPORT.md](REPORT.md) for full methodology, results, and discussion.
