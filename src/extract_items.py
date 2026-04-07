"""
Extract clean item names from LLM responses using regex heuristics.
Then re-save processed results with improved extraction.
"""

import json
import re
from pathlib import Path
from collections import Counter

WORKSPACE = Path("/workspaces/underrated-novelty-nlp-claude")
RESULTS_DIR = WORKSPACE / "results"


def extract_item_name(response):
    """Extract the primary item/entity name from a response.

    Strategy:
    1. Check for quoted items (most reliable)
    2. Check for bold items (**item**)
    3. Take first line and clean it
    """
    if not response:
        return ""

    text = response.strip()

    # Strategy 1: Look for quoted entity in first line
    first_line = text.split("\n")[0]
    # Match "Item Name" or 'Item Name'
    quoted = re.findall(r'["""]([^"""]+)["""]', first_line)
    if not quoted:
        quoted = re.findall(r"[''']([^''']+)[''']", first_line)
    if quoted:
        return quoted[0].strip()

    # Strategy 2: Look for **bold** text
    bold = re.findall(r'\*\*([^*]+)\*\*', first_line)
    if bold:
        return bold[0].strip()

    # Strategy 3: Parse first line heuristically
    line = first_line.strip()

    # Remove leading patterns like "I would say " etc.
    prefixes = [
        r"^(I would say|I'd say|My pick is|My answer is|I think)\s+",
        r"^(The most underrated\s+\w+\s+is)\s+",
        r"^(One of the most underrated\s+\w+\s+is)\s+",
    ]
    for pat in prefixes:
        line = re.sub(pat, "", line, flags=re.IGNORECASE)

    # Take text before common explanation markers
    for marker in [" is the most ", " is one of ", " because ", " - because", " — ", " – ", ". It ", ". This ", ". Despite "]:
        idx = line.lower().find(marker.lower())
        if idx > 3:  # at least some text before marker
            line = line[:idx]
            break

    # If still has a period and text before it, take before period
    if ". " in line:
        before_period = line[:line.index(". ")]
        if len(before_period) > 3:
            line = before_period

    # Clean up
    line = line.strip().strip("*").strip('"').strip("'").strip(".").strip(",").strip()

    # If too long, it probably has explanation — try to cut at "is"
    if len(line) > 80:
        is_match = re.search(r'^(.{5,60}?)\s+is\s+', line)
        if is_match:
            line = is_match.group(1)

    # Final truncation
    if len(line) > 80:
        line = line[:80]

    return line.strip()


def normalize_item(item):
    """Normalize item name for comparison."""
    item = item.lower().strip()
    # Remove common articles and punctuation
    item = re.sub(r'^(the|a|an)\s+', '', item)
    item = re.sub(r'[^\w\s]', '', item)
    item = re.sub(r'\s+', ' ', item).strip()
    return item


def main():
    # Load raw responses
    cache_file = RESULTS_DIR / "raw_responses_cache.jsonl"
    results = []
    with open(cache_file) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Processing {len(results)} responses...")

    # Extract items
    for r in results:
        r["extracted_item"] = extract_item_name(r["response"])
        r["normalized_item"] = normalize_item(r["extracted_item"])

    # Save processed results
    with open(RESULTS_DIR / "processed_responses.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print some examples for validation
    print("\n=== Extraction Examples ===")
    models = sorted(set(r["model"] for r in results))
    sample_prompts = [
        "What is the most underrated movie of all time?",
        "What is the most underrated programming language?",
        "What is the most underrated cuisine?",
        "What is the best movie of all time?",
        "What is the highest-grossing movie of all time?"
    ]

    for prompt in sample_prompts:
        print(f"\n--- {prompt} ---")
        for model in models:
            items = [r["normalized_item"] for r in results
                    if r["prompt_text"] == prompt and r["model"] == model]
            counter = Counter(items)
            n = len(items)
            print(f"  {model} ({n} samples):")
            for item, count in counter.most_common(5):
                print(f"    {item}: {count}/{n} ({100*count/n:.0f}%)")

    # Overall stats
    print(f"\n\n=== Overall Extraction Stats ===")
    items = [r["extracted_item"] for r in results]
    empty = sum(1 for i in items if not i)
    long = sum(1 for i in items if len(i) > 60)
    print(f"Empty extractions: {empty}/{len(items)}")
    print(f"Long extractions (>60 chars): {long}/{len(items)}")
    avg_len = sum(len(i) for i in items) / len(items)
    print(f"Average extraction length: {avg_len:.1f} chars")


if __name__ == "__main__":
    main()
