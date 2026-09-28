"""Inference wrapper for fine-tuned seq2seq summarizers (BART / T5 / PEGASUS).

    from src.summarizer import Summarizer
    s = Summarizer("bart-base-cnn-summarizer")          # local folder or Hugging Face Hub id
    s.summarize(article)                                # one text -> str
    s.summarize([a1, a2], batch_size=16)               # many texts -> list[str]
    s.summarize_long(paper_text)                        # research papers / very long documents

Long documents are handled by hierarchical ("map-reduce") summarization: the text is split into
sentence-aligned chunks that fit the encoder, every chunk is summarized, the partial summaries are
concatenated and summarized again.
"""
from __future__ import annotations

import json
import os
import time

from .preprocessing import clean_article, split_sentences

__all__ = ["Summarizer", "DEFAULT_GENERATION", "chunk_sentences", "is_t5_like"]

# Decoding settings used for CNN/DailyMail (same as the reference BART-large-CNN configuration).
DEFAULT_GENERATION = dict(num_beams=4, max_new_tokens=142, min_new_tokens=56, length_penalty=2.0,
                          no_repeat_ngram_size=3, early_stopping=True)
T5_TYPES = {"t5", "mt5", "umt5", "longt5"}


def is_t5_like(model_type: str) -> bool:
    return model_type in T5_TYPES


def _split_long_sentence(sentence: str, count_tokens, max_tokens: int) -> list[str]:
    pieces, current = [], []
    for word in sentence.split():
        current.append(word)
        if len(current) > 1 and count_tokens(" ".join(current)) > max_tokens:
            current.pop()
            pieces.append(" ".join(current))
            current = [word]
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_sentences(sentences: list[str], count_tokens, max_tokens: int, overlap: int = 1) -> list[list[str]]:
    """Greedy sentence packing into chunks of <= max_tokens tokens with `overlap` sentences of context."""
    max_tokens = max(2, int(max_tokens))
    units = []                                   # every unit costs n + 1 tokens (1 = separator margin)
    for s in sentences:
        n = count_tokens(s)
        if n + 1 > max_tokens:
            units.extend((p, count_tokens(p)) for p in _split_long_sentence(s, count_tokens, max_tokens - 1))
        else:
            units.append((s, n))
    chunks, current, current_len = [], [], 0
    for s, n in units:
        if current and current_len + n + 1 > max_tokens:
            chunks.append([u for u, _ in current])
            keep = current[-overlap:] if overlap > 0 else []
            keep_len = sum(k + 1 for _, k in keep)
            if keep_len + n + 1 > max_tokens:
                keep, keep_len = [], 0
            current, current_len = list(keep), keep_len
        current.append((s, n))
        current_len += n + 1
    if current:
        chunks.append([u for u, _ in current])
    return chunks


class Summarizer:
    def __init__(self, model_name_or_path: str, device: str | None = None, fp16: bool | None = None,
                 max_source_len: int | None = None, generation: dict | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).eval()
        meta_path = os.path.join(model_name_or_path, "summarizer_config.json")
        self.meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        model_type = self.model.config.model_type
        self.prefix = self.meta.get("prefix", "summarize: " if is_t5_like(model_type) else "")
        self.clean_input = self.meta.get("clean_input", True)
        position_limit = getattr(self.model.config, "max_position_embeddings", None) or 1024
        self.max_source_len = int(max_source_len or self.meta.get("max_source_len") or min(position_limit, 1024))
        self.generation = dict(DEFAULT_GENERATION)
        self.generation.update(self.meta.get("generation", {}))
        self.generation.update(generation or {})
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if fp16 is None:
            fp16 = self.device.startswith("cuda") and not is_t5_like(model_type)
        if fp16:
            self.model.half()
        self.model.to(self.device)

    # ----------------------------------------------------------------------------------------- utils
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def prepare(self, text: str) -> str:
        return clean_article(text) if self.clean_input else " ".join(str(text).split())

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _resolve(self, overrides: dict) -> dict:
        gen = dict(self.generation)
        gen.update({k: v for k, v in overrides.items() if v is not None})
        if gen.get("max_new_tokens") is not None and gen.get("min_new_tokens") is not None:
            gen["min_new_tokens"] = max(0, min(gen["min_new_tokens"], gen["max_new_tokens"] - 1))
        return gen

    def _adaptive_lengths(self, n_input_tokens: int) -> dict:
        """Shorter inputs get proportionally shorter summaries (CNN/DM settings assume ~700 words)."""
        max_new = min(self.generation.get("max_new_tokens", 142), max(24, int(0.55 * n_input_tokens)))
        min_new = min(self.generation.get("min_new_tokens", 56), max(8, int(0.15 * n_input_tokens)))
        return {"max_new_tokens": max_new, "min_new_tokens": min(min_new, max_new - 1)}

    # ------------------------------------------------------------------------------------ generation
    def _generate(self, prepared: list[str], batch_size: int, gen: dict, progress_every: int = 0) -> list[str]:
        torch = self._torch
        order = sorted(range(len(prepared)), key=lambda i: -len(prepared[i]))
        outputs: list[str | None] = [None] * len(prepared)
        t0, n_batches = time.time(), (len(order) + batch_size - 1) // max(1, batch_size)
        for b, start in enumerate(range(0, len(order), batch_size), 1):
            idx = order[start:start + batch_size]
            enc = self.tokenizer([self.prefix + prepared[i] for i in idx], max_length=self.max_source_len,
                                 truncation=True, padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(**enc, **gen)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for i, text in zip(idx, decoded):
                outputs[i] = text.strip()
            if progress_every and (b % progress_every == 0 or b == n_batches):
                elapsed = time.time() - t0
                eta = elapsed / b * (n_batches - b)
                print(f"  generated {min(start + batch_size, len(order)):>6}/{len(order)} "
                      f"| {elapsed / 60:5.1f} min elapsed | ETA {eta / 60:5.1f} min", flush=True)
        return outputs  # type: ignore[return-value]

    def summarize(self, texts, batch_size: int = 16, adaptive_length: bool | None = None,
                  progress_every: int = 0, **generation_overrides):
        """Summarize one text (returns str) or a list of texts (returns list[str]).

        adaptive_length (default: True for a single text, False for lists) scales the summary length
        with the input length, which gives better results for short, non-news inputs.
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if adaptive_length is None:
            adaptive_length = single
        prepared = [self.prepare(t) for t in items]
        if not adaptive_length:
            outs = self._generate(prepared, batch_size, self._resolve(generation_overrides), progress_every)
        else:
            outs = []
            for p in prepared:
                n_tokens = min(self.count_tokens(p), self.max_source_len)
                overrides = {**self._adaptive_lengths(n_tokens), **generation_overrides}
                outs.extend(self._generate([p], 1, self._resolve(overrides)))
        return outs[0] if single else outs

    def summarize_long(self, text: str, chunk_tokens: int | None = None, overlap_sentences: int = 1,
                       chunk_generation: dict | None = None, final_generation: dict | None = None,
                       max_rounds: int = 2, batch_size: int = 8) -> dict:
        """Hierarchical summarization for documents longer than the encoder window."""
        prepared = self.prepare(text)
        budget = (chunk_tokens or self.max_source_len) - self.count_tokens(self.prefix) - 8
        n_tokens = self.count_tokens(prepared)
        if n_tokens <= budget:
            return {"summary": self.summarize(prepared, **(final_generation or {})), "input_tokens": n_tokens,
                    "chunks": 1, "rounds": 0, "partial_summaries": []}
        current, rounds, partial_log, n_chunks_first = prepared, 0, [], None
        while rounds < max_rounds and self.count_tokens(current) > budget:
            chunks = chunk_sentences(split_sentences(current), self.count_tokens, budget, overlap_sentences)
            n_chunks_first = n_chunks_first or len(chunks)
            chunk_gen = chunk_generation or {"max_new_tokens": 96, "min_new_tokens": 24}
            partials = self.summarize([" ".join(c) for c in chunks], batch_size=batch_size,
                                      adaptive_length=False, **chunk_gen)
            partial_log.append(partials)
            current = " ".join(p.replace("\n", " ") for p in partials)
            rounds += 1
        final = self.summarize(current, adaptive_length=False, **(final_generation or {}))
        return {"summary": final, "input_tokens": n_tokens, "chunks": n_chunks_first, "rounds": rounds,
                "partial_summaries": partial_log}
