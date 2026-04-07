# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "'Most Underrated' as a Novelty Metric." Resources include 14 academic papers, 1 custom dataset, and 3 code repositories supporting experimentation on LLM output novelty and diversity.

## Papers
Total papers downloaded: 14

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Generative Monoculture in LLMs | Wu, Black, Chandrasekaran | 2024 | papers/2407.02209_generative_monoculture.pdf | Defines monoculture, RLHF reduces diversity |
| 2 | Creative Homogeneity Across LLMs | Wenger, Kenett | 2025 | papers/2501.19361_creative_homogeneity_across_llms.pdf | Cross-model homogeneity, AUT/FF/DAT tests |
| 3 | Magic, Madness, Heaven, Sin | Dhingra | 2026 | papers/2604.01504_llm_output_diversity.pdf | Unified framework for output variation |
| 4 | LLM Outputs: Similarity, Diversity, Bias | Multiple | 2025 | papers/2505.09056_llm_outputs_similarity_diversity_bias.pdf | 3M texts, 12 LLMs analysis |
| 5 | CreativityPrism | Multiple | 2025 | papers/2510.20091_creativity_prism.pdf | Holistic creativity evaluation framework |
| 6 | LiveIdeaBench | Multiple | 2024 | papers/2412.17596_liveideabench.pdf | Scientific creativity benchmark |
| 7 | Divergent Creativity in Humans and LLMs | Multiple | 2024 | papers/2405.13012_divergent_creativity_humans_llms.pdf | DAT comparison, 100K humans |
| 8 | Benchmarking LM Creativity (Code) | Multiple | 2024 | papers/2407.09007_benchmarking_lm_creativity.pdf | Denial Prompting, NeoGauge |
| 9 | LLM Novelty via Semantic Similarity | Davydov et al. | 2025 | papers/2510.27313_llm_novelty_semantic_similarity.pdf | Semantic novelty framework |
| 10 | Multi-Novelty | Multiple | 2025 | papers/2502.12700_multi_novelty.pdf | Multi-view brainstorming for diversity |
| 11 | Human Creativity in Age of LLMs | Multiple | 2024 | papers/2410.03703_human_creativity_age_llms.pdf | Randomized experiments |
| 12 | Evaluating Diversity in NLG | Tevet, Berant | 2020 | papers/2004.02990_evaluating_diversity_nlg.pdf | Diversity metric evaluation framework |
| 13 | LLMs as Recommender Systems: Popularity Bias | Lichtenberg et al. | 2024 | papers/llm_recommender_popularity_bias.pdf | Popularity bias metrics |
| 14 | Barriers to Diversity in LLM Ideas | Multiple | 2026 | papers/2602.20408_barriers_diversity_llm_ideas.pdf | Barriers and mitigations |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets created/identified: 1 custom + references to external datasets

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Most Underrated Benchmark | Custom | 96 prompts | Novelty evaluation | datasets/underrated_benchmark/ | 80 underrated + 16 control prompts |

External datasets referenced in literature (not downloaded, available on demand):
- **Goodreads book reviews**: Used by GeMo paper, available via Goodreads API
- **CodeContests**: Google DeepMind competitive programming dataset
- **AUT/FF/DAT human data**: Available at OSF (osf.io/u3yv4, osf.io/7p5mt, osf.io/kbeq6)
- **SmolLM corpus chunks**: Available at HuggingFace (stai-tuebingen/faiss-smollm)

See datasets/README.md for detailed descriptions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| GeMo | github.com/GeMoLLM/GeMo | Generative monoculture measurement | code/GeMo/ | Dispersion metrics framework |
| Divergent | github.com/lechmazur/divergent | Divergent thinking benchmark | code/divergent/ | Multi-LLM comparison data |
| LiveIdeaBench | github.com/x66ccff/liveideabench | Scientific creativity evaluation | code/liveideabench/ | Nature Communications paper |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder tool with diligent mode for 3 search queries
2. Web searches on Google Scholar, arXiv, Semantic Scholar
3. Targeted searches for popularity bias, semantic diversity, and output homogeneity
4. GitHub API searches for implementation code

### Selection Criteria
- Papers directly addressing LLM output diversity/homogeneity
- Papers on creativity evaluation with quantitative metrics
- Papers on novelty measurement methodology
- Papers on popularity bias in LLM outputs
- Code repos with reusable evaluation frameworks

### Challenges Encountered
- No existing dataset specifically tests "most underrated" convergence — addressed by creating custom benchmark
- The research topic spans multiple subfields (creativity, diversity, bias, novelty) with different terminology
- Some papers use "diversity" to mean different things in different contexts (addressed by [3] framework)

### Gaps and Workarounds
- **No human baseline for "underrated" questions**: Will need crowdsourcing in experiment phase
- **No established "underratedness" ground truth**: Can proxy via popularity metrics (box office, sales, etc.)
- **Limited code for opinion diversity measurement**: GeMo framework can be adapted

## Recommendations for Experiment Design

### Primary Dataset
**Most Underrated Novelty Benchmark** — 96 prompts across 8 categories designed to test whether LLMs converge on the same "underrated" answers.

### Baseline Methods
1. **Multi-model comparison**: Query ≥5 LLMs from different families
2. **Intra-model sampling**: ≥20 responses per prompt per model at varying temperatures
3. **Human survey**: Collect comparable human responses via crowdsourcing
4. **Control prompts**: "Best X" and "Most popular X" as convergence calibration

### Evaluation Metrics
1. **Convergence rate**: Fraction of identical/near-identical answers across models and samples
2. **Semantic diversity**: Mean pairwise cosine distance of response embeddings (all-MiniLM-L6-v2)
3. **Response entropy**: Shannon entropy over unique extracted items
4. **Popularity correlation**: Named items' actual popularity vs. how "underrated" they truly are
5. **Inter-vs-intra model diversity**: Compare within-model vs. across-model response diversity

### Code to Adapt/Reuse
1. **GeMo**: Dispersion metrics, attribute extraction pipeline, comparison framework
2. **sentence-transformers**: all-MiniLM-L6-v2 for embedding-based diversity measurement
3. **LiveIdeaBench**: Multi-dimensional evaluation methodology
