#!/usr/bin/env python3
import os
from pathlib import Path

from openai import OpenAI


def summarize_log(log_path: Path) -> str:
    text = log_path.read_text(errors="ignore")
    if not text.strip():
        return "No log output available."

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a CI assistant. "
                "Summarize the root cause of this log failure in one short sentence (<=50 characters)."
                "Answer only with the sentence, no other text.",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=80,
    )

    return resp.choices[0].message.content.strip()


def main(max_chars: int = 4000):
    root = Path("outputs")
    for d in root.glob("*"):
        log = d / "run_trace.log"
        if log.exists():
            summary = summarize_log(log)
            (d / "error_summary.txt").write_text(summary + "\n")

            text = log.read_text(errors="ignore")
            if len(text) > max_chars:
                text = text[-max_chars:]

            trace_md = f"<details><summary>Show run_trace.log</summary>\n\n```text\n{text}\n```\n\n</details>\n"
            (d / "error_trace.md").write_text(trace_md)


if __name__ == "__main__":
    main()
