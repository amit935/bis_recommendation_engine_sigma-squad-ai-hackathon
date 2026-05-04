"""
generate_training_data.py
Generates vague query training pairs for all BIS standards using Groq API.

Usage:
    python generate_training_data.py --groq_key YOUR_KEY
    python generate_training_data.py --groq_key YOUR_KEY --existing gemini_data.json
"""

import argparse
import json
import os
import pickle
import time
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────

CHUNKS_PATH    = "bis_chunks.pkl"
OUTPUT_PATH    = "training_data.json"
GROQ_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
BATCH_SIZE     = 15      # Increased batch size since we have 30K TPM now
SLEEP_BETWEEN  = 3       # Reduced sleep since rate limit is 5x higher
MAX_RETRIES    = 3       # Max retries per batch on failure

# ── Load Existing Data ────────────────────────────────────────────────────────

def load_existing(path: str) -> dict:
    """Load already generated pairs (from Gemini or previous run)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    existing = {item["standard_id"]: item for item in data}
    print(f"Loaded {len(existing)} existing standards from {path}")
    return existing


def load_chunks(path: str) -> list:
    """Load BIS standard chunks from cached pickle."""
    with open(path, "rb") as f:
        docs = pickle.load(f)
    standards = [
        {"standard_id": d["standard_id"], "standard_title": d["title"]}
        for d in docs
        if d["standard_id"] != "Unknown"
    ]
    # Deduplicate by standard_id
    seen = set()
    unique = []
    for s in standards:
        if s["standard_id"] not in seen:
            seen.add(s["standard_id"])
            unique.append(s)
    print(f"Loaded {len(unique)} unique standards from chunks")
    return unique


# ── Groq Generation ───────────────────────────────────────────────────────────

def build_prompt(standards: list) -> str:
    """Build prompt for a batch of standards."""
    standards_list = "\n".join(
        f'{i+1}. {s["standard_id"]} — {s["standard_title"]}'
        for i, s in enumerate(standards)
    )

    return f"""You are building a training dataset for an AI model that helps Indian MSE owners find BIS compliance standards.

For each BIS standard below, generate exactly 5 vague non-technical descriptions that a small Indian factory or workshop owner might use to describe their product.

Rules:
- Use simple everyday English only
- Do NOT use the exact technical words from the standard title
- Each description must be 1 sentence, maximum 15 words
- Cover different angles: material, appearance, use case, process, customer type
- Think like a small business owner, not an engineer

Return ONLY a valid JSON array. No explanation, no markdown, no extra text before or after.
Strictly follow this exact format:

[
  {{
    "standard_id": "IS 269: 1989",
    "standard_title": "33 grade ordinary Portland cement",
    "vague_descriptions": [
      "we make the grey powder used for binding bricks in construction",
      "our factory produces the white mixing material for building houses",
      "I supply the fine dust that hardens when mixed with water for walls",
      "we manufacture the binding material builders use for walls and floors",
      "we make the fine powder that sets hard when mixed with water"
    ]
  }}
]

HERE ARE THE STANDARDS TO PROCESS:
{standards_list}"""


def generate_batch(client: Groq, standards: list, batch_num: int) -> list:
    """Send one batch to Groq and parse response with retry + backoff."""

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  Sending batch {batch_num} (attempt {attempt}, {len(standards)} standards)...")

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": build_prompt(standards)}],
                max_tokens=8192,
                temperature=0.7,
            )

            raw = response.choices[0].message.content.strip()

            # Clean any markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            print(f"  Batch {batch_num}: got {len(parsed)} standards")
            return parsed

        except json.JSONDecodeError as e:
            print(f"  Batch {batch_num} JSON parse error: {e}")
            print(f"  Raw response preview: {raw[:200]}")
        except Exception as e:
            err_msg = str(e)
            print(f"  Batch {batch_num} API error: {err_msg[:120]}")

            # Rate limit: wait longer before retry
            if "429" in err_msg or "rate" in err_msg.lower():
                wait = 15 * attempt
                print(f"  Rate limited. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

        # Backoff between retries
        if attempt < MAX_RETRIES:
            wait = 5 * attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  Batch {batch_num} FAILED after {MAX_RETRIES} attempts. Skipping.")
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate BIS training data via Groq")
    parser.add_argument("--groq_key",  required=True, help="Groq API key")
    parser.add_argument("--existing",  default=None,  help="Path to existing JSON (e.g. gemini_data.json)")
    parser.add_argument("--output",    default=OUTPUT_PATH, help="Output JSON path")
    parser.add_argument("--chunks",    default=CHUNKS_PATH, help="Path to bis_chunks.pkl")
    args = parser.parse_args()

    # Load existing data
    existing = load_existing(args.existing)

    # Load all standards from chunks
    all_standards = load_chunks(args.chunks)

    # Filter out already generated ones
    remaining = [
        s for s in all_standards
        if s["standard_id"] not in existing
    ]
    print(f"Standards to generate: {len(remaining)} (skipping {len(existing)} already done)")

    if not remaining:
        print("All standards already generated. Nothing to do.")
        return

    # Init Groq client
    client = Groq(api_key=args.groq_key)

    # Process in batches
    all_results = list(existing.values())  # Start with existing data
    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(remaining), BATCH_SIZE):
        batch      = remaining[i:i + BATCH_SIZE]
        batch_num  = (i // BATCH_SIZE) + 1

        print(f"\nBatch {batch_num}/{total_batches}")
        result = generate_batch(client, batch, batch_num)

        if result:
            all_results.extend(result)
            # Save progress after every batch -- safe against crashes
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Progress saved: {len(all_results)} total standards so far")
        else:
            print(f"  Batch {batch_num} failed after all retries. Moving on.")

        if i + BATCH_SIZE < len(remaining):
            time.sleep(SLEEP_BETWEEN)

    # Final save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"Total standards     : {len(all_results)}")
    print(f"Total training pairs: {len(all_results) * 5}")
    print(f"Saved to            : {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()