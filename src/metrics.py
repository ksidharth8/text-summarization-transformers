"""Evaluation metrics for summarization.

* ROUGE-1/2/L/Lsum F1 (x100) via Google's `rouge-score` package with Porter stemming.
  Sentences are put on separate lines so that ROUGE-Lsum is the summary-level LCS used in papers.
* 95% bootstrap confidence intervals and paired bootstrap tests between two systems.
* Abstractiveness: % of summary n-grams that never occur in the source article.
* Length statistics.
"""
from __future__ import annotations

import numpy as np

from .preprocessing import split_sentences, word_tokenize

__all__ = ["ROUGE_TYPES", "rouge_per_sample", "bootstrap_ci", "paired_bootstrap", "novel_ngram_ratio",
           "length_stats", "evaluate_system"]

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")


def _one_sentence_per_line(text: str) -> str:
    return "\n".join(split_sentences(text))


def rouge_per_sample(predictions: list[str], references: list[str], use_stemmer: bool = True,
                     rouge_types=ROUGE_TYPES) -> dict[str, np.ndarray]:
    """Per-example ROUGE F1 scores (x100)."""
    from rouge_score import rouge_scorer

    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions vs {len(references)} references")
    scorer = rouge_scorer.RougeScorer(list(rouge_types), use_stemmer=use_stemmer)
    scores = {t: np.zeros(len(predictions), dtype=np.float64) for t in rouge_types}
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        s = scorer.score(_one_sentence_per_line(ref), _one_sentence_per_line(pred))   # (target, prediction)
        for t in rouge_types:
            scores[t][i] = 100.0 * s[t].fmeasure
    return scores


def bootstrap_ci(values, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, values.size, values.size)].mean() for _ in range(n_resamples)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(a, b, n_resamples: int = 1000, seed: int = 0) -> dict:
    """Mean difference a-b with 95% CI and the fraction of resamples where a <= b (one-sided p-value)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return {"diff": float("nan"), "ci": [float("nan"), float("nan")], "p_value": float("nan")}
    d = a - b
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_resamples)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 2), "ci": [round(float(lo), 2), round(float(hi), 2)],
            "p_value": round(float((means <= 0).mean()), 4)}


def _ngrams(tokens: list[str], n: int) -> set:
    return set(zip(*[tokens[i:] for i in range(n)]))


def novel_ngram_ratio(sources: list[str], summaries: list[str], ns=(1, 2, 3, 4)) -> dict[str, float]:
    """Average % of summary n-grams that do not appear in the source (0 = purely extractive)."""
    ratios = {n: [] for n in ns}
    for src, summ in zip(sources, summaries):
        src_tokens, summ_tokens = word_tokenize(src), word_tokenize(summ)
        for n in ns:
            summ_ngrams = _ngrams(summ_tokens, n)
            if summ_ngrams:
                ratios[n].append(len(summ_ngrams - _ngrams(src_tokens, n)) / len(summ_ngrams))
    return {str(n): (round(100.0 * float(np.mean(v)), 2) if v else float("nan")) for n, v in ratios.items()}


def length_stats(texts: list[str]) -> dict[str, float]:
    words = np.array([len(t.split()) for t in texts]) if texts else np.zeros(1)
    sents = np.array([len(split_sentences(t)) for t in texts]) if texts else np.zeros(1)
    return {"avg_words": round(float(words.mean()), 1), "median_words": float(np.median(words)),
            "avg_sentences": round(float(sents.mean()), 2)}


def evaluate_system(predictions: list[str], references: list[str], sources: list[str] | None = None,
                    novelty_idx=None, with_ci: bool = True) -> tuple[dict, dict]:
    """ROUGE (+CI), length and (optionally, on a subset) abstractiveness for one system."""
    per_sample = rouge_per_sample(predictions, references)
    result: dict = {"n": len(predictions)}
    for t, v in per_sample.items():
        result[t] = round(float(v.mean()), 2)
        if with_ci:
            lo, hi = bootstrap_ci(v)
            result[t + "_ci"] = [round(lo, 2), round(hi, 2)]
    result.update(length_stats(predictions))
    if sources is not None:
        idx = range(len(predictions)) if novelty_idx is None else novelty_idx
        result["novel_ngrams"] = novel_ngram_ratio([sources[i] for i in idx], [predictions[i] for i in idx])
    return result, per_sample
