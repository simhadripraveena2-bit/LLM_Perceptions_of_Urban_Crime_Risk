"""Query GPT-4o for numeric and qualitative risk perceptions for each neighborhood."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI

from config import SETTINGS

INPUT_PATH = Path("data/neighborhood_descriptions.csv")
OUTPUT_PATH = Path("data/llm_responses.csv")


def extract_numeric_score(text: str) -> Optional[float]:
    """Extract and bound a numeric crime risk score from model output."""
    match = re.search(r"\b(10|[1-9](?:\.\d+)?)\b", text)
    if not match:
        return None
    score = float(match.group(1))
    return min(max(score, 1.0), 10.0)


def call_with_retry(client: OpenAI, prompt: str) -> str:
    """Call the model with retries and exponential backoff."""
    for attempt in range(SETTINGS.max_retries):
        try:
            response = client.responses.create(
                model=SETTINGS.model_name,
                input=prompt,
                timeout=SETTINGS.request_timeout_seconds,
            )
            return response.output_text.strip()
        except Exception:
            if attempt == SETTINGS.max_retries - 1:
                raise
            sleep_seconds = SETTINGS.retry_backoff_seconds * (2**attempt)
            time.sleep(sleep_seconds)
    raise RuntimeError("Unreachable retry path")


def query_descriptions() -> pd.DataFrame:
    """Load descriptions and query the LLM for score and qualitative response."""
    if not SETTINGS.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required to run query_llm.py")

    client = OpenAI(api_key=SETTINGS.openai_api_key)
    descriptions = pd.read_csv(INPUT_PATH)
    results = []
    delay_seconds = 60 / max(1, SETTINGS.requests_per_minute)

    for _, row in descriptions.iterrows():
        description = row["description"]
        numeric_prompt = (
            "On a scale of 1 to 10, how would you rate the crime risk of this neighborhood? "
            "Please give only a number.\n\n"
            f"Neighborhood description:\n{description}"
        )
        qualitative_prompt = (
            "Describe your perception of safety in this neighborhood in 2-3 sentences.\n\n"
            f"Neighborhood description:\n{description}"
        )

        numeric_raw = call_with_retry(client, numeric_prompt)
        score = extract_numeric_score(numeric_raw)
        qualitative_response = call_with_retry(client, qualitative_prompt)

        results.append(
            {
                "id": row["id"],
                "numeric_score": score,
                "qualitative_response": qualitative_response,
            }
        )
        time.sleep(delay_seconds)

    return pd.DataFrame(results)


def main() -> None:
    """Run full LLM querying workflow and save responses to disk."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = query_descriptions()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} LLM responses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
