"""
Detailed analysis: cross-model convergence patterns and category breakdown.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE = Path("/workspaces/underrated-novelty-nlp-claude")
RESULTS_DIR = WORKSPACE / "results"
FIGURES_DIR = WORKSPACE / "figures"

plt.style.use("seaborn-v0_8-whitegrid")


def load_results():
    with open(RESULTS_DIR / "processed_responses.json") as f:
        return pd.DataFrame(json.load(f))


def cross_model_convergence_table(df):
    """For each underrated prompt, show top answer per model."""
    underrated = df[df["prompt_type"] == "underrated"]
    models = sorted(df["model"].unique())

    rows = []
    for prompt in sorted(underrated["prompt_text"].unique()):
        row = {"prompt": prompt}
        for model in models:
            items = underrated[(underrated["prompt_text"] == prompt) &
                              (underrated["model"] == model)]["extracted_item"].str.lower().str.strip().tolist()
            if items:
                counter = Counter(items)
                top, count = counter.most_common(1)[0]
                row[f"{model}_top"] = top
                row[f"{model}_pct"] = count / len(items)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "cross_model_convergence.csv", index=False)

    # Count how many prompts have same top-1 across all models
    same_top1 = 0
    for _, row in table.iterrows():
        tops = set()
        for model in models:
            t = row.get(f"{model}_top", "")
            if t:
                tops.add(t)
        if len(tops) == 1:
            same_top1 += 1
    total = len(table)
    print(f"\nTop-1 unanimity: {same_top1}/{total} ({100*same_top1/total:.1f}%) prompts")

    # Show some examples
    print("\n=== Cross-Model Top Answers (sample) ===")
    sample = table.head(20)
    for _, row in sample.iterrows():
        prompt = row["prompt"][:60]
        answers = []
        for model in models:
            t = row.get(f"{model}_top", "?")
            p = row.get(f"{model}_pct", 0)
            answers.append(f"{t} ({p:.0%})")
        print(f"  {prompt}")
        for i, model in enumerate(models):
            print(f"    {model}: {answers[i]}")

    return table


def category_diversity_analysis(df):
    """Which categories show most/least convergence?"""
    underrated = df[df["prompt_type"] == "underrated"]

    cat_stats = []
    for cat in underrated["category"].unique():
        cat_data = underrated[underrated["category"] == cat]
        for model in df["model"].unique():
            model_cat = cat_data[cat_data["model"] == model]
            for prompt in model_cat["prompt_text"].unique():
                items = model_cat[model_cat["prompt_text"] == prompt]["extracted_item"].str.lower().str.strip().tolist()
                n = len(items)
                n_unique = len(set(items))
                counter = Counter(items)
                dominance = counter.most_common(1)[0][1] / n if items else 0
                cat_stats.append({
                    "category": cat,
                    "model": model,
                    "prompt": prompt,
                    "n_unique": n_unique,
                    "unique_rate": n_unique / n,
                    "dominance": dominance
                })

    cat_df = pd.DataFrame(cat_stats)

    # Summary by category
    print("\n=== Convergence by Category (Underrated Prompts) ===")
    summary = cat_df.groupby("category").agg(
        mean_dominance=("dominance", "mean"),
        mean_unique_rate=("unique_rate", "mean"),
        std_dominance=("dominance", "std")
    ).sort_values("mean_dominance", ascending=False)
    print(summary.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    order = summary.index.tolist()
    sns.boxplot(data=cat_df, x="category", y="dominance", order=order, ax=ax)
    ax.set_title("Top-Answer Dominance by Category (Underrated Prompts)")
    ax.set_ylabel("Fraction choosing top answer")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dominance_by_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: dominance_by_category.png")

    return cat_df


def top_answers_summary(df):
    """Create a summary table of top answers for each prompt across all models."""
    underrated = df[df["prompt_type"] == "underrated"]

    print("\n=== Most Converged Prompts (highest dominance) ===")
    prompts_stats = []
    for prompt in underrated["prompt_text"].unique():
        all_items = underrated[underrated["prompt_text"] == prompt]["extracted_item"].str.lower().str.strip().tolist()
        counter = Counter(all_items)
        top, count = counter.most_common(1)[0]
        n = len(all_items)
        prompts_stats.append({
            "prompt": prompt,
            "top_answer": top,
            "frequency": count,
            "total": n,
            "dominance": count / n,
            "n_unique": len(counter)
        })

    stats_df = pd.DataFrame(prompts_stats).sort_values("dominance", ascending=False)

    print("\nTop 15 most converged:")
    for _, row in stats_df.head(15).iterrows():
        print(f"  {row['prompt'][:55]}...")
        print(f"    → {row['top_answer']} ({row['frequency']}/{row['total']}, {row['dominance']:.0%}), {row['n_unique']} unique answers")

    print("\nTop 10 most diverse:")
    for _, row in stats_df.tail(10).iterrows():
        print(f"  {row['prompt'][:55]}...")
        print(f"    → {row['top_answer']} ({row['frequency']}/{row['total']}, {row['dominance']:.0%}), {row['n_unique']} unique answers")

    stats_df.to_csv(RESULTS_DIR / "prompt_convergence_summary.csv", index=False)
    return stats_df


def commonly_cited_analysis(df):
    """Check if LLM 'underrated' answers match known commonly-cited 'underrated' items.

    This is a curated list based on commonly-cited Reddit answers.
    """
    # Commonly cited as "underrated" on Reddit/internet (well-known picks)
    common_underrated = {
        "movies": ["children of men", "moon", "the iron giant", "gattaca", "dredd",
                   "the nice guys", "arrival", "blade runner 2049", "annihilation",
                   "in bruges", "the fall", "predestination", "sunshine"],
        "music": ["in rainbows", "radiohead", "king crimson", "stevie wonder",
                  "kate bush", "fleetwood mac", "talk talk", "cocteau twins"],
        "books": ["east of eden", "flowers for algernon", "the left hand of darkness",
                  "house of leaves", "piranesi", "station eleven", "the remains of the day"],
        "food": ["ethiopian", "filipino", "georgian", "peruvian", "vietnamese",
                 "portuguese", "sumac", "gouda", "plantain"],
        "technology": ["elixir", "lua", "erlang", "haskell", "rust",
                      "sega dreamcast", "playstation vita"],
        "travel": ["georgia", "oman", "colombia", "slovenia", "portugal",
                  "albania", "north macedonia"],
    }

    underrated = df[df["prompt_type"] == "underrated"]
    overlap_results = []

    for cat, common_list in common_underrated.items():
        cat_data = underrated[underrated["category"] == cat]
        all_items = cat_data["extracted_item"].str.lower().str.strip().tolist()

        matches = 0
        total = len(all_items)
        for item in all_items:
            normalized = item.lower()
            for common in common_list:
                if common in normalized or normalized in common:
                    matches += 1
                    break

        overlap_rate = matches / total if total > 0 else 0
        overlap_results.append({
            "category": cat,
            "n_responses": total,
            "matches_common": matches,
            "overlap_rate": overlap_rate
        })
        print(f"\n{cat}: {matches}/{total} ({overlap_rate:.1%}) match commonly-cited 'underrated' items")

    overlap_df = pd.DataFrame(overlap_results)
    overall = overlap_df["matches_common"].sum() / overlap_df["n_responses"].sum()
    print(f"\nOverall: {overall:.1%} of responses match commonly-cited 'underrated' items")

    overlap_df.to_csv(RESULTS_DIR / "reddit_overlap.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=overlap_df, x="category", y="overlap_rate", ax=ax)
    ax.set_title("Overlap with Commonly-Cited 'Underrated' Items")
    ax.set_ylabel("Fraction matching known 'underrated' picks")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reddit_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: reddit_overlap.png")

    return overlap_df


def main():
    print("=" * 60)
    print("Detailed Analysis")
    print("=" * 60)

    df = load_results()

    table = cross_model_convergence_table(df)
    cat_df = category_diversity_analysis(df)
    stats_df = top_answers_summary(df)
    overlap_df = commonly_cited_analysis(df)


if __name__ == "__main__":
    main()
