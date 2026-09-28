"""Gradio web demo:  python app/app.py   (then open http://127.0.0.1:7860)

Model folder: models/bart-base-cnn-summarizer (default) or the SUMM_MODEL_DIR environment variable.
Set SUMM_SHARE=1 to get a temporary public link (useful on Kaggle / Colab).
"""
from __future__ import annotations

import glob
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import gradio as gr  # noqa: E402

from src.extractive import lead_k, textrank  # noqa: E402

MODEL_DIR = os.environ.get("SUMM_MODEL_DIR", os.path.join(ROOT, "models", "bart-base-cnn-summarizer"))
METHODS = ["Abstractive (fine-tuned BART)", "TextRank (extractive)", "LEAD-3 (extractive)"]
_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        from src.summarizer import Summarizer
        _summarizer = Summarizer(MODEL_DIR)
    return _summarizer


def summarize(text: str, method: str, num_beams: int = 4):
    text = (text or "").strip()
    if len(text.split()) < 20:
        return "Please paste at least a few sentences (20+ words).", ""
    t0 = time.time()
    if method == METHODS[1]:
        return textrank(text, 3), f"TextRank: top-3 central sentences | {time.time() - t0:.2f}s"
    if method == METHODS[2]:
        return lead_k(text, 3), f"LEAD-3: first three sentences | {time.time() - t0:.2f}s"
    s = get_summarizer()
    n_tok = s.count_tokens(s.prepare(text))
    if n_tok > s.max_source_len - s.count_tokens(s.prefix) - 8:
        out = s.summarize_long(text, final_generation={"num_beams": int(num_beams)})
        info = f"{n_tok} tokens -> long-document mode ({out['chunks']} chunks, {out['rounds']} round(s))"
        summary = out["summary"]
    else:
        summary, info = s.summarize(text, num_beams=int(num_beams)), f"{n_tok} tokens"
    return summary, f"{info} | {time.time() - t0:.1f}s on {s.device}"


def build_demo():
    examples = []
    for path in sorted(glob.glob(os.path.join(ROOT, "test_cases", "*.txt"))):
        with open(path, encoding="utf-8") as f:
            examples.append([f.read().strip(), METHODS[0], 4])
    return gr.Interface(
        fn=summarize,
        inputs=[gr.Textbox(lines=14, label="Article / paper text"),
                gr.Radio(METHODS, value=METHODS[0], label="Method"),
                gr.Slider(1, 8, value=4, step=1, label="Beam size (abstractive)")],
        outputs=[gr.Textbox(lines=6, label="Summary"), gr.Textbox(label="Details")],
        examples=examples or None,
        title="Text Summarization with Transformers",
        description="BART-base fine-tuned on CNN/DailyMail vs. extractive baselines. "
                    "Long documents are summarized hierarchically (chunk -> partial summaries -> final summary).",
    )


if __name__ == "__main__":
    build_demo().launch(share=os.environ.get("SUMM_SHARE") == "1")
