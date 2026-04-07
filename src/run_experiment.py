"""
Experiment: Query LLMs with 'most underrated' prompts and measure convergence.

Tests whether LLMs converge on the same "obvious" underrated answers,
indicating a lack of genuine novelty/taste.
"""

import json
import os
import sys
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Config
NUM_SAMPLES = 20  # samples per prompt per model
TEMPERATURE = 1.0
MAX_TOKENS = 150
MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4o"]

# System prompt to get concise, single-item answers
SYSTEM_PROMPT = (
    "You are answering a question about what is most underrated in a given category. "
    "Give a single, specific answer (one item only). Start your response with the name "
    "of the item, then briefly explain why in 1-2 sentences. Do not hedge or list multiple items."
)

SYSTEM_PROMPT_CONTROL = (
    "You are answering a question. Give a single, specific answer (one item only). "
    "Start your response with the name of the item, then briefly explain why in 1-2 sentences. "
    "Do not hedge or list multiple items."
)

WORKSPACE = Path("/workspaces/underrated-novelty-nlp-claude")
RESULTS_DIR = WORKSPACE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_prompts():
    """Load the benchmark prompts."""
    with open(WORKSPACE / "datasets" / "underrated_benchmark" / "prompts.json") as f:
        data = json.load(f)

    prompts = []
    # Underrated prompts
    for category, prompt_list in data["underrated_prompts"].items():
        for prompt_text in prompt_list:
            prompts.append({
                "text": prompt_text,
                "category": category,
                "type": "underrated"
            })
    # Control prompts
    for control_type, prompt_list in data["control_prompts"].items():
        for prompt_text in prompt_list:
            prompts.append({
                "text": prompt_text,
                "category": control_type,
                "type": control_type
            })
    return prompts


def query_model(client, model, prompt_text, prompt_type, sample_id):
    """Query a single model with a single prompt."""
    sys_prompt = SYSTEM_PROMPT if prompt_type == "underrated" else SYSTEM_PROMPT_CONTROL

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED + sample_id  # vary seed per sample for diversity
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Error querying {model}: {e}")
        time.sleep(2)
        return None


def run_all_queries(prompts, models, num_samples):
    """Run all queries across prompts, models, and samples."""
    client = OpenAI()
    results = []

    total = len(prompts) * len(models) * num_samples
    print(f"Total API calls planned: {total}")

    # Cache file to allow resumption
    cache_file = RESULTS_DIR / "raw_responses_cache.jsonl"
    cached = set()
    if cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                r = json.loads(line)
                key = f"{r['model']}|{r['prompt_text']}|{r['sample_id']}"
                cached.add(key)
                results.append(r)
        print(f"Loaded {len(cached)} cached responses")

    remaining = []
    for prompt in prompts:
        for model in models:
            for sid in range(num_samples):
                key = f"{model}|{prompt['text']}|{sid}"
                if key not in cached:
                    remaining.append((prompt, model, sid))

    print(f"Remaining API calls: {len(remaining)}")

    if not remaining:
        return results

    # Use ThreadPoolExecutor for parallel API calls
    with open(cache_file, "a") as cache_f:
        # Process in batches to manage rate limits
        batch_size = 50
        pbar = tqdm(total=len(remaining), desc="Querying models")

        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                for prompt, model, sid in batch:
                    future = executor.submit(
                        query_model, client, model, prompt["text"], prompt["type"], sid
                    )
                    futures[future] = (prompt, model, sid)

                for future in as_completed(futures):
                    prompt, model, sid = futures[future]
                    response_text = future.result()

                    if response_text is not None:
                        result = {
                            "model": model,
                            "prompt_text": prompt["text"],
                            "category": prompt["category"],
                            "prompt_type": prompt["type"],
                            "sample_id": sid,
                            "response": response_text,
                            "timestamp": datetime.now().isoformat()
                        }
                        results.append(result)
                        cache_f.write(json.dumps(result) + "\n")
                        cache_f.flush()

                    pbar.update(1)

            # Small delay between batches for rate limiting
            time.sleep(0.5)

        pbar.close()

    return results


def extract_item_name(response):
    """Extract the primary item name from a response.
    Takes the first line or first sentence as the item name.
    """
    if not response:
        return ""
    # Take first line
    first_line = response.split("\n")[0].strip()
    # Remove common prefixes
    for prefix in ["The most underrated", "I would say", "My pick is", "I think"]:
        if first_line.lower().startswith(prefix.lower()):
            first_line = first_line[len(prefix):].strip().lstrip(":")
    # Take up to first period or dash
    for sep in [".", " - ", " – ", " — "]:
        if sep in first_line:
            first_line = first_line[:first_line.index(sep)]
    # Clean up
    first_line = first_line.strip().strip("*").strip('"').strip("'").strip()
    # Truncate if too long (likely includes explanation)
    if len(first_line) > 100:
        first_line = first_line[:100]
    return first_line


def save_results(results):
    """Save processed results."""
    # Save raw results
    with open(RESULTS_DIR / "raw_responses.json", "w") as f:
        json.dump(results, f, indent=2)

    # Extract items and save processed results
    processed = []
    for r in results:
        item = extract_item_name(r["response"])
        processed.append({
            **r,
            "extracted_item": item
        })

    with open(RESULTS_DIR / "processed_responses.json", "w") as f:
        json.dump(processed, f, indent=2)

    print(f"Saved {len(processed)} processed responses")
    return processed


def main():
    print("=" * 60)
    print("'Most Underrated' Novelty Experiment")
    print("=" * 60)
    print(f"Models: {MODELS}")
    print(f"Samples per prompt: {NUM_SAMPLES}")
    print(f"Temperature: {TEMPERATURE}")
    print()

    # Load prompts
    prompts = load_prompts()
    print(f"Loaded {len(prompts)} prompts ({sum(1 for p in prompts if p['type']=='underrated')} underrated, "
          f"{sum(1 for p in prompts if p['type']!='underrated')} control)")

    # Run queries
    results = run_all_queries(prompts, MODELS, NUM_SAMPLES)
    print(f"\nCollected {len(results)} total responses")

    # Process and save
    processed = save_results(results)

    # Quick summary
    print("\n--- Quick Summary ---")
    for model in MODELS:
        model_results = [r for r in processed if r["model"] == model]
        print(f"\n{model}: {len(model_results)} responses")
        # Show convergence for a sample prompt
        sample_prompt = "What is the most underrated movie of all time?"
        sample_responses = [r["extracted_item"] for r in model_results
                          if r["prompt_text"] == sample_prompt]
        if sample_responses:
            from collections import Counter
            counts = Counter(sample_responses)
            print(f"  '{sample_prompt}':")
            for item, count in counts.most_common(5):
                print(f"    {item}: {count}/{len(sample_responses)}")


if __name__ == "__main__":
    main()
