# Literature Review: 'Most Underrated' as a Novelty Metric

## Research Area Overview

This literature review examines the intersection of LLM output diversity, creativity evaluation, and novelty measurement. The central research question is whether LLMs can generate genuinely novel responses—particularly when asked to name the "most underrated" item in a category—or whether they converge on widely-known "underrated" answers, revealing a lack of deeper novelty reasoning.

The literature reveals a strong and growing body of evidence that LLMs suffer from **generative monoculture**: their outputs are significantly less diverse than human-generated content, both within a single model (intra-model) and across different models (inter-model). This phenomenon is directly relevant to our hypothesis, as it predicts that LLMs asked for "underrated" items will converge on the same popular-but-perceived-as-underrated answers.

---

## Key Papers

### 1. Generative Monoculture in Large Language Models
- **Authors**: Fan Wu, Emily Black, Varun Chandrasekaran (UIUC, Barnard)
- **Year**: 2024
- **Source**: arXiv:2407.02209
- **Key Contribution**: Introduces the concept of *generative monoculture*—the narrowing of LLM output distributions relative to training data. Formally defined as: Dispersion(h(x)|x~P_gen) < Dispersion(h(x)|x~P_src).
- **Methodology**: Compared diversity of LLM-generated vs. human-generated book reviews (Goodreads, 742 books) and code solutions (CodeContests, 100 problems). Measured sentiment, topic, wording, code fingerprints, and algorithmic diversity.
- **Key Results**:
  - LLM-generated book reviews are overwhelmingly positive (GPT-3.5/4: 100% in top sentiment range), while human reviews span the full sentiment spectrum.
  - Code solutions show narrower algorithmic diversity with near-100% plagiarism scores between generated solutions.
  - RLHF alignment is the primary driver—pre-trained Llama-2 has much more diverse output than RLHF-tuned Llama-2-chat.
  - **Naive mitigations (temperature, top-p, prompts) are insufficient** to restore diversity.
- **Datasets Used**: Goodreads book reviews, CodeContests
- **Code Available**: https://github.com/GeMoLLM/GeMo
- **Relevance**: Directly supports hypothesis. If LLMs exhibit monoculture in sentiment and algorithms, they will likely also converge on the same "underrated" answers.

### 2. We're Different, We're the Same: Creative Homogeneity Across LLMs
- **Authors**: Emily Wenger, Yoed Kenett (Duke, Technion)
- **Year**: 2025
- **Source**: arXiv:2501.19361
- **Key Contribution**: Demonstrates that creative homogeneity exists not just within a single LLM but **across different LLM families**. Even models from different companies (Meta, Google, OpenAI, Mistral, etc.) produce remarkably similar creative outputs.
- **Methodology**: Applied three standardized creativity tests (AUT, Forward Flow, DAT) to 22 LLMs and 102 humans. Measured population-level response variability using sentence embedding cosine distances.
- **Key Results**:
  - LLMs match or slightly outperform humans on individual originality scores.
  - But LLM responses are **far less variable** than human responses (effect sizes 1.4-2.2, all p < 1e-10).
  - Even controlling for model family, LLM responses cluster tightly in embedding space while human responses are dispersed.
  - Creative system prompts slightly increase variability but do not close the gap.
- **Datasets Used**: Custom IRB-approved human study (Prolific, 102 participants), plus public AUT/FF/DAT datasets from OSF.
- **Evaluation Metrics**: GloVe embeddings for originality; all-MiniLM-L6-v2 sentence embeddings for variability; Welch's t-test.
- **Relevance**: Critical finding—even using different LLMs won't produce diverse "underrated" answers. The homogeneity is systemic.

### 3. Magic, Madness, Heaven, Sin: LLM Output Diversity is Everything, Everywhere, All at Once
- **Authors**: Harnoor Dhingra (Microsoft)
- **Year**: 2026
- **Source**: arXiv:2604.01504
- **Key Contribution**: Proposes a unified framework for understanding LLM output variation across four normative contexts: epistemic (factuality), interactional (user utility), societal (representation), and safety (robustness).
- **Methodology**: Survey/framework paper organizing the fragmented diversity literature.
- **Key Insight**: The value of output variation depends on context. For "most underrated" prompts, we're in the **interactional context** where heterogeneity is desirable (Magic), but LLMs may default to epistemic convergence (Madness) or safety-driven homogeneity (Heaven).
- **Relevance**: Provides theoretical framing for why LLMs might converge on "safe" underrated answers.

### 4. LLM Generation Novelty Through the Lens of Semantic Similarity
- **Authors**: Philipp Davydov et al. (Tübingen AI Center)
- **Year**: 2025/2026
- **Source**: arXiv:2510.27313
- **Key Contribution**: Proposes a three-stage semantic novelty framework: (1) retrieve semantically similar training samples, (2) rerank at varying sequence lengths, (3) calibrate against human novelty baseline.
- **Methodology**: Applied to SmolLM model family against full pretraining corpus (~20TB). Uses embedding-based similarity rather than n-gram overlap.
- **Key Results**:
  - Models reuse pretraining data over much longer sequences than previously reported.
  - Novelty varies systematically by task domain.
  - Instruction tuning increases novelty (beyond just style changes).
  - N-gram overlap methods miss paraphrased content; embedding similarity is more reliable.
- **Code/Data**: Released ~20TB of corpus chunks at HuggingFace.
- **Relevance**: Provides methodology for measuring whether LLM "underrated" answers are truly novel or semantically reproduce training data patterns.

### 5. Large Language Models as Recommender Systems: A Study of Popularity Bias
- **Authors**: Lichtenberg, Buchholz, Schwöbel (Amazon)
- **Year**: 2024
- **Source**: Amazon Science
- **Key Contribution**: Studies popularity bias in LLM-based recommender systems. Introduces a principled popularity bias metric framework.
- **Key Results**:
  - LLM recommenders actually exhibit *less* popularity bias than traditional collaborative filtering.
  - Prompting can further reduce bias ("recommend niche content").
  - However, LLMs still favor items more represented in training data.
- **Relevance**: Directly relevant—when LLMs name "underrated" items, they may still favor popular items that appear frequently in training data as "underrated" (a meta-popularity bias).

### 6. Divergent Creativity in Humans and Large Language Models
- **Authors**: Multiple authors
- **Year**: 2024
- **Source**: arXiv:2405.13012
- **Key Contribution**: Compares divergent creativity between LLMs and 100,000 humans using the Divergent Association Task.
- **Key Results**: LLMs surpass average human performance but fall short of highly creative humans. Individual scores can be high while population diversity remains low.
- **Relevance**: Confirms the individual-vs-population creativity paradox relevant to our hypothesis.

### 7. Examining and Addressing Barriers to Diversity in LLM-Generated Ideas
- **Authors**: Multiple authors
- **Year**: 2026
- **Source**: arXiv:2602.20408
- **Relevance**: Examines what barriers exist to LLM idea diversity and proposes mitigation strategies.

### 8. Diminished Diversity-of-Thought in a Standard Large Language Model
- **Authors**: Multiple authors (PMC)
- **Year**: 2024
- **Key Finding**: 99.7% of GPT-3.5 runs (1,027/1,030) chose the same answer to survey questions, with only 0.3% offering different responses. Demonstrates extreme convergence in opinion-like tasks.
- **Relevance**: Strongest evidence that LLMs will converge on the same "underrated" answers.

---

## Common Methodologies

### Measuring Output Diversity
1. **Semantic similarity** (cosine distance of sentence embeddings): Used in [2, 4]. all-MiniLM-L6-v2 is standard.
2. **Entropy/dispersion metrics**: Used in [1]. Applied to extracted attributes (sentiment, topic, algorithms).
3. **Distribution comparison**: Welch's t-test, KL divergence between source and generated distributions.
4. **Embedding visualization**: t-SNE of response embeddings to visualize clustering.
5. **Lexical overlap**: Word overlap counts, Jaccard similarity of response tokens.

### Eliciting Responses
- Multiple samples per prompt (10-100 generations per query)
- Varying temperature and top-p parameters
- Multiple LLMs from different families
- Human baselines via crowd-sourcing (Prolific, MTurk)

---

## Standard Baselines
- **Temperature/top-p variation**: Standard diversity intervention, shown to be insufficient [1, 2]
- **Persona prompting**: "Write as if you are X" — modest improvement [1]
- **Creative system prompts**: Slight increase in variability but doesn't close gap [2]
- **Human responses**: Gold standard for diversity comparison

---

## Evaluation Metrics
- **Pairwise cosine similarity** of sentence embeddings (lower = more diverse)
- **Entropy** over categorical attributes (higher = more diverse)
- **Standard deviation** of continuous attributes
- **Jaccard index** for set-based attributes
- **n-gram novelty** (proportion of novel n-grams vs. training data)
- **Semantic novelty** (embedding distance from nearest training sample) [4]

---

## Datasets in the Literature
- **Goodreads book reviews**: Used in [1] for sentiment diversity analysis
- **CodeContests**: Used in [1] for code diversity analysis
- **AUT/FF/DAT human creativity data**: Used in [2], available on OSF
- **SmolLM pretraining corpus**: Used in [4], available on HuggingFace
- **MovieLens**: Used in [5] for recommendation popularity bias

---

## Gaps and Opportunities

1. **No existing benchmark for "underrated" novelty**: No dataset specifically tests whether LLMs converge on the same "underrated" items. This is our primary contribution.
2. **Subjective vs. objective novelty**: Most work measures diversity relative to training data or other outputs. Our "most underrated" prompt naturally tests subjective novelty.
3. **Category-specific analysis**: No prior work examines how novelty varies across different knowledge domains (movies vs. science vs. food).
4. **Cross-model convergence on opinions**: While [2] shows creative homogeneity, no work specifically tests opinion convergence on subjective preference questions.
5. **Popularity bias meets novelty**: The intersection of recommendation popularity bias [5] and creative diversity [1, 2] is unexplored.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Custom "Most Underrated" Benchmark** (created): 80 underrated prompts + 16 control prompts across 8 categories.
2. **Human survey data**: Collect human responses to the same prompts via crowdsourcing for ground truth diversity comparison.

### Recommended Baselines
1. Multiple LLMs from different families (GPT-4, Claude, Llama, Gemini, Mistral)
2. Multiple samples per prompt per model (N≥20)
3. Temperature/top-p variations as ablation
4. Human responses as gold standard

### Recommended Metrics
1. **Response convergence rate**: How often do different LLMs give the same answer?
2. **Semantic diversity**: Pairwise cosine similarity of response embeddings (all-MiniLM-L6-v2)
3. **Response entropy**: Shannon entropy over unique answers
4. **Popularity correlation**: Correlate named items with their actual popularity (e.g., box office, sales data)
5. **Novelty score**: Fraction of responses that are genuinely obscure vs. commonly cited as "underrated"

### Methodological Considerations
- Use structured prompts that force concise answers (e.g., "Name one item only")
- Run each prompt multiple times per model (≥20) to measure intra-model diversity
- Compare intra-model vs. inter-model diversity
- Consider temperature as an independent variable
- Validate "underratedness" against popularity metrics (box office, Spotify plays, etc.)
