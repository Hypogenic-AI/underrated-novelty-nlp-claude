"""
Analysis: Measure convergence and semantic diversity of LLM 'underrated' responses.

Computes:
1. Unique answer rate per prompt per model
2. Semantic diversity via sentence embeddings
3. Inter-model overlap
4. Comparison between underrated, best, and factual prompts
5. Statistical tests
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

WORKSPACE = Path("/workspaces/underrated-novelty-nlp-claude")
RESULTS_DIR = WORKSPACE / "results"
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")


def load_results():
    """Load processed responses."""
    with open(RESULTS_DIR / "processed_responses.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} responses")
    print(f"Models: {df['model'].unique().tolist()}")
    print(f"Prompt types: {df['prompt_type'].unique().tolist()}")
    return df


def compute_unique_answer_rate(df):
    """Compute the proportion of unique answers per prompt per model."""
    results = []
    for (model, prompt_text), group in df.groupby(["model", "prompt_text"]):
        items = group["extracted_item"].str.lower().str.strip().tolist()
        n_total = len(items)
        n_unique = len(set(items))
        unique_rate = n_unique / n_total if n_total > 0 else 0
        prompt_type = group["prompt_type"].iloc[0]
        category = group["category"].iloc[0]

        # Also compute entropy
        counter = Counter(items)
        probs = np.array(list(counter.values())) / n_total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Most common answer and its frequency
        most_common, mc_count = counter.most_common(1)[0]
        dominance = mc_count / n_total

        results.append({
            "model": model,
            "prompt_text": prompt_text,
            "prompt_type": prompt_type,
            "category": category,
            "n_samples": n_total,
            "n_unique": n_unique,
            "unique_rate": unique_rate,
            "entropy": entropy,
            "most_common": most_common,
            "dominance": dominance  # fraction of most common answer
        })

    return pd.DataFrame(results)


def compute_semantic_diversity(df, model_name):
    """Compute pairwise semantic diversity for each prompt using sentence embeddings."""
    print(f"Loading sentence transformer model...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = []
    prompts = df["prompt_text"].unique()

    for prompt_text in tqdm(prompts, desc=f"Semantic diversity ({model_name})"):
        group = df[(df["prompt_text"] == prompt_text)]
        responses = group["response"].tolist()

        if len(responses) < 2:
            continue

        embeddings = st_model.encode(responses, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)

        # Mean pairwise similarity (excluding diagonal)
        n = len(responses)
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        mean_sim = sim_matrix[mask].mean()
        std_sim = sim_matrix[mask].std()

        prompt_type = group["prompt_type"].iloc[0]
        category = group["category"].iloc[0]
        model = group["model"].iloc[0]

        results.append({
            "model": model,
            "prompt_text": prompt_text,
            "prompt_type": prompt_type,
            "category": category,
            "mean_pairwise_similarity": mean_sim,
            "std_pairwise_similarity": std_sim,
            "semantic_diversity": 1 - mean_sim,  # higher = more diverse
            "n_responses": n
        })

    return pd.DataFrame(results)


def compute_intermodel_overlap(df):
    """Compute overlap of top answers across models for each prompt."""
    models = df["model"].unique()
    results = []

    for prompt_text in df["prompt_text"].unique():
        prompt_data = df[df["prompt_text"] == prompt_text]
        prompt_type = prompt_data["prompt_type"].iloc[0]
        category = prompt_data["category"].iloc[0]

        # Get top-5 answers per model
        model_tops = {}
        for model in models:
            model_group = prompt_data[prompt_data["model"] == model]
            items = model_group["extracted_item"].str.lower().str.strip().tolist()
            counter = Counter(items)
            model_tops[model] = set(item for item, _ in counter.most_common(5))

        # Compute pairwise Jaccard similarity
        jaccard_scores = []
        model_list = list(models)
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                set_a = model_tops[model_list[i]]
                set_b = model_tops[model_list[j]]
                if len(set_a | set_b) > 0:
                    jaccard = len(set_a & set_b) / len(set_a | set_b)
                    jaccard_scores.append(jaccard)

        # Also check: does the #1 answer match across models?
        top1_answers = []
        for model in models:
            model_group = prompt_data[prompt_data["model"] == model]
            items = model_group["extracted_item"].str.lower().str.strip().tolist()
            if items:
                top1_answers.append(Counter(items).most_common(1)[0][0])
        top1_match = len(set(top1_answers)) == 1  # all models agree on #1

        results.append({
            "prompt_text": prompt_text,
            "prompt_type": prompt_type,
            "category": category,
            "mean_jaccard": np.mean(jaccard_scores) if jaccard_scores else 0,
            "top1_unanimous": top1_match,
            "n_unique_top1": len(set(top1_answers))
        })

    return pd.DataFrame(results)


def run_statistical_tests(unique_df):
    """Run statistical tests comparing prompt types."""
    results = {}

    prompt_types = unique_df["prompt_type"].unique()
    metrics = ["unique_rate", "entropy", "dominance"]

    for metric in metrics:
        underrated = unique_df[unique_df["prompt_type"] == "underrated"][metric].values
        popular = unique_df[unique_df["prompt_type"] == "popular_consensus"][metric].values
        factual = unique_df[unique_df["prompt_type"] == "factual_baseline"][metric].values

        # Underrated vs Popular consensus
        t_stat, p_val = stats.ttest_ind(underrated, popular, equal_var=False)
        cohens_d = (underrated.mean() - popular.mean()) / np.sqrt(
            (underrated.std()**2 + popular.std()**2) / 2
        )
        results[f"{metric}_underrated_vs_popular"] = {
            "t_stat": t_stat, "p_value": p_val, "cohens_d": cohens_d,
            "mean_underrated": underrated.mean(), "mean_popular": popular.mean()
        }

        # Underrated vs Factual
        t_stat, p_val = stats.ttest_ind(underrated, factual, equal_var=False)
        cohens_d = (underrated.mean() - factual.mean()) / np.sqrt(
            (underrated.std()**2 + factual.std()**2) / 2
        )
        results[f"{metric}_underrated_vs_factual"] = {
            "t_stat": t_stat, "p_value": p_val, "cohens_d": cohens_d,
            "mean_underrated": underrated.mean(), "mean_factual": factual.mean()
        }

    return results


def plot_unique_rate_by_type(unique_df):
    """Plot unique answer rate by prompt type."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(["unique_rate", "entropy", "dominance"]):
        ax = axes[i]
        sns.boxplot(data=unique_df, x="prompt_type", y=metric, ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Prompt Type")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "diversity_by_prompt_type.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: diversity_by_prompt_type.png")


def plot_semantic_diversity(sem_df):
    """Plot semantic diversity by prompt type and category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By prompt type
    sns.boxplot(data=sem_df, x="prompt_type", y="semantic_diversity", ax=axes[0])
    axes[0].set_title("Semantic Diversity by Prompt Type")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)

    # By category (underrated only)
    underrated_sem = sem_df[sem_df["prompt_type"] == "underrated"]
    if not underrated_sem.empty:
        sns.boxplot(data=underrated_sem, x="category", y="semantic_diversity", ax=axes[1])
        axes[1].set_title("Semantic Diversity by Category (Underrated)")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "semantic_diversity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: semantic_diversity.png")


def plot_intermodel_overlap(overlap_df):
    """Plot inter-model overlap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=overlap_df, x="prompt_type", y="mean_jaccard", ax=axes[0])
    axes[0].set_title("Inter-Model Overlap (Jaccard) by Prompt Type")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)

    # Top-1 unanimity rate by type
    unanimity = overlap_df.groupby("prompt_type")["top1_unanimous"].mean().reset_index()
    sns.barplot(data=unanimity, x="prompt_type", y="top1_unanimous", ax=axes[1])
    axes[1].set_title("Top-1 Answer Unanimity Rate by Prompt Type")
    axes[1].set_ylabel("Fraction with same #1 answer")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "intermodel_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: intermodel_overlap.png")


def plot_convergence_heatmap(df):
    """Heatmap of dominance (most common answer %) by category and model."""
    unique_df = compute_unique_answer_rate(df)
    underrated = unique_df[unique_df["prompt_type"] == "underrated"]

    pivot = underrated.pivot_table(values="dominance", index="category", columns="model", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Mean Dominance (Top Answer Frequency) by Category and Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "convergence_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: convergence_heatmap.png")


def plot_model_comparison(unique_df):
    """Compare models on diversity metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    underrated = unique_df[unique_df["prompt_type"] == "underrated"]

    for i, metric in enumerate(["unique_rate", "entropy", "dominance"]):
        ax = axes[i]
        sns.boxplot(data=underrated, x="model", y=metric, ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} by Model (Underrated Prompts)")
        ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: model_comparison.png")


def show_example_convergence(df, n_prompts=5):
    """Print example convergence patterns for illustration."""
    print("\n" + "=" * 70)
    print("EXAMPLE CONVERGENCE PATTERNS")
    print("=" * 70)

    underrated = df[df["prompt_type"] == "underrated"]
    prompts = underrated["prompt_text"].unique()[:n_prompts]
    models = df["model"].unique()

    for prompt in prompts:
        print(f"\n--- {prompt} ---")
        for model in models:
            items = underrated[
                (underrated["prompt_text"] == prompt) &
                (underrated["model"] == model)
            ]["extracted_item"].str.lower().str.strip().tolist()
            counter = Counter(items)
            top3 = counter.most_common(3)
            total = len(items)
            print(f"  {model}:")
            for item, count in top3:
                print(f"    {item}: {count}/{total} ({100*count/total:.0f}%)")


def main():
    print("=" * 60)
    print("Analysis: 'Most Underrated' Novelty Experiment")
    print("=" * 60)

    df = load_results()

    # 1. Unique answer rate
    print("\n--- Computing unique answer rates ---")
    unique_df = compute_unique_answer_rate(df)
    unique_df.to_csv(RESULTS_DIR / "unique_answer_rates.csv", index=False)

    # Summary stats
    for ptype in unique_df["prompt_type"].unique():
        subset = unique_df[unique_df["prompt_type"] == ptype]
        print(f"\n{ptype}:")
        print(f"  Unique rate: {subset['unique_rate'].mean():.3f} ± {subset['unique_rate'].std():.3f}")
        print(f"  Entropy:     {subset['entropy'].mean():.3f} ± {subset['entropy'].std():.3f}")
        print(f"  Dominance:   {subset['dominance'].mean():.3f} ± {subset['dominance'].std():.3f}")

    # 2. Semantic diversity (per model to save memory)
    print("\n--- Computing semantic diversity ---")
    sem_dfs = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        sem = compute_semantic_diversity(model_df, model)
        sem_dfs.append(sem)
    sem_df = pd.concat(sem_dfs, ignore_index=True)
    sem_df.to_csv(RESULTS_DIR / "semantic_diversity.csv", index=False)

    for ptype in sem_df["prompt_type"].unique():
        subset = sem_df[sem_df["prompt_type"] == ptype]
        print(f"\n{ptype} (semantic diversity):")
        print(f"  Mean: {subset['semantic_diversity'].mean():.4f} ± {subset['semantic_diversity'].std():.4f}")

    # 3. Inter-model overlap
    print("\n--- Computing inter-model overlap ---")
    overlap_df = compute_intermodel_overlap(df)
    overlap_df.to_csv(RESULTS_DIR / "intermodel_overlap.csv", index=False)

    for ptype in overlap_df["prompt_type"].unique():
        subset = overlap_df[overlap_df["prompt_type"] == ptype]
        print(f"\n{ptype} (inter-model overlap):")
        print(f"  Jaccard:  {subset['mean_jaccard'].mean():.3f} ± {subset['mean_jaccard'].std():.3f}")
        print(f"  Top-1 unanimity: {subset['top1_unanimous'].mean():.1%}")

    # 4. Statistical tests
    print("\n--- Statistical tests ---")
    stat_results = run_statistical_tests(unique_df)
    for test_name, res in stat_results.items():
        print(f"\n{test_name}:")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)

    # 5. Visualizations
    print("\n--- Generating plots ---")
    plot_unique_rate_by_type(unique_df)
    plot_semantic_diversity(sem_df)
    plot_intermodel_overlap(overlap_df)
    plot_convergence_heatmap(df)
    plot_model_comparison(unique_df)

    # 6. Example convergence patterns
    show_example_convergence(df)

    print("\n" + "=" * 60)
    print("Analysis complete. Results saved to results/ and figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
