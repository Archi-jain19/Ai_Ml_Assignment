"""
Interactive manual testing script for the facet-scoring pipeline.

Allows testing custom conversation snippets against the 399 facet catalogue.

Usage:
    # 1. Interactive prompt mode:
    python scripts/score_custom.py

    # 2. Direct argument mode:
    python scripts/score_custom.py "I spent three days debugging a memory leak until the bug was fixed."
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import run_pipeline
from src.config import TOP_K

logging.basicConfig(level=logging.WARNING)  # Clean output for interactive testing

def main():
    if len(sys.argv) > 1:
        conversation = " ".join(sys.argv[1:])
    else:
        print("\n" + "=" * 60)
        print("FACET SCORING PIPELINE — INTERACTIVE MANUAL TEST")
        print("=" * 60)
        print("Enter a conversation snippet to evaluate (or 'exit' to quit):\n")
        try:
            conversation = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not conversation or conversation.lower() == "exit":
        print("No input provided. Exiting.")
        sys.exit(0)

    result = run_pipeline(
        conversation=conversation,
        conversation_id="manual_test_01",
    )

    print("\n" + "=" * 80)
    print(f"Conversation: \"{conversation}\"\n")

    for i, res in enumerate(result.get("results", []), 1):
        facet = res["facet"]
        status = res["status"]
        score = res["score"]
        conf = res["confidence"]
        reason = res["reason"]

        status_tag = f"[{status.upper()}]"
        if status == "scored":
            print(f"{i:2d}. {facet:<36} {status_tag:<24} Score: {score}/5 (Conf: {conf:.2f})")
        else:
            print(f"{i:2d}. {facet:<36} {status_tag:<24} Score: null (Conf: {conf:.2f})")
        print(f"    Reason: {reason}\n")

    print("=" * 80)

if __name__ == "__main__":
    main()
