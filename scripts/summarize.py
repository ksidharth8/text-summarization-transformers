#!/usr/bin/env python
"""Summarize text from the command line with the fine-tuned model and/or the extractive baselines.

Examples
--------
  python scripts/summarize.py --file test_cases/01_news_electric_buses.txt
  python scripts/summarize.py --file test_cases/*.txt --method abstractive
  python scripts/summarize.py --text "Paste an article here ..." --method all
  cat article.txt | python scripts/summarize.py

Documents longer than the model's input window (512 tokens) are summarized hierarchically
(chunks -> partial summaries -> final summary) automatically; --long forces that mode.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.extractive import lead_k, textrank  # noqa: E402

DEFAULT_MODEL = os.environ.get("SUMM_MODEL_DIR", os.path.join(ROOT, "models", "bart-base-cnn-summarizer"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", default=DEFAULT_MODEL, help="trained model folder or Hugging Face Hub id")
    ap.add_argument("--text", help="text to summarize")
    ap.add_argument("--file", nargs="*", help="one or more .txt files")
    ap.add_argument("--method", choices=["abstractive", "textrank", "lead3", "all"], default="all")
    ap.add_argument("--long", action="store_true", help="force hierarchical long-document mode")
    ap.add_argument("--num_beams", type=int)
    ap.add_argument("--max_new_tokens", type=int)
    ap.add_argument("--min_new_tokens", type=int)
    ap.add_argument("--device", help="cpu / cuda (default: auto)")
    args = ap.parse_args()

    docs: list[tuple[str, str]] = []
    if args.text:
        docs.append(("text", args.text))
    for path in args.file or []:
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    if not docs and not sys.stdin.isatty():
        docs.append(("stdin", sys.stdin.read()))
    if not docs:
        ap.error("give --text, --file or pipe text on stdin")

    summarizer = None
    if args.method in ("abstractive", "all"):
        from src.summarizer import Summarizer
        parent = os.path.dirname(args.model_dir.rstrip("/"))
        looks_local = args.model_dir.startswith((".", "/", "~")) or (parent and os.path.isdir(parent))
        if not os.path.isdir(args.model_dir) and looks_local:
            sys.exit(f"model folder not found: {args.model_dir}\n-> unzip bart-base-cnn-summarizer.zip into models/ "
                     "or pass --model_dir")
        t0 = time.time()
        summarizer = Summarizer(args.model_dir, device=args.device)
        print(f"[loaded {args.model_dir} on {summarizer.device} in {time.time() - t0:.1f}s]\n")
    overrides = {k: getattr(args, k) for k in ("num_beams", "max_new_tokens", "min_new_tokens")
                 if getattr(args, k) is not None}

    for name, text in docs:
        print("=" * 100 + f"\n{name}  ({len(text.split())} words)\n" + "=" * 100)
        if summarizer is not None:
            t0 = time.time()
            budget = summarizer.max_source_len - summarizer.count_tokens(summarizer.prefix) - 8
            n_tok = summarizer.count_tokens(summarizer.prepare(text))
            if args.long or n_tok > budget:
                out = summarizer.summarize_long(text, final_generation=overrides or None)
                info = f"long-document mode: {n_tok} tokens -> {out['chunks']} chunks, {out['rounds']} round(s)"
                summary = out["summary"]
            else:
                summary, info = summarizer.summarize(text, **overrides), f"{n_tok} tokens"
            print(f"\n[ABSTRACTIVE] ({info}, {time.time() - t0:.1f}s)\n{summary}")
        if args.method in ("textrank", "all"):
            print(f"\n[TEXTRANK]\n{textrank(text, 3)}")
        if args.method in ("lead3", "all"):
            print(f"\n[LEAD-3]\n{lead_k(text, 3)}")
        print()


if __name__ == "__main__":
    main()
