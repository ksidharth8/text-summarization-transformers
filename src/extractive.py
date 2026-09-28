"""Extractive summarization methods.

* LEAD-k          first k sentences (exploits the "inverted pyramid" style of news writing)
* TextRank        PageRank over a sentence-similarity graph (Mihalcea & Tarau, 2004)
* Embedding MMR   transformer sentence embeddings (MiniLM) -> centroid relevance + Maximal
                  Marginal Relevance for diversity (Carbonell & Goldstein, 1998)

All methods return the selected sentences in their original document order, one per line.
"""
from __future__ import annotations

import numpy as np

from .preprocessing import split_sentences, word_tokenize

__all__ = ["lead_k", "textrank", "textrank_scores", "mmr_select", "EmbeddingExtractor", "STOPWORDS"]

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most my
    myself no nor not now of off on once only or other our ours ourselves out over own same she should so
    some such than that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves s t said says also one two new year years mr mrs ms""".split()
)


def _candidate_indices(sentences: list[str], min_words: int) -> list[int]:
    idx = [i for i, s in enumerate(sentences) if len(s.split()) >= min_words]
    return idx or list(range(len(sentences)))


def lead_k(text: str, k: int = 3, min_words: int = 4) -> str:
    """First k sentences (fragments shorter than `min_words` words are skipped)."""
    sentences = split_sentences(text)
    idx = _candidate_indices(sentences, min_words)[:k]
    return "\n".join(sentences[i] for i in idx)


def textrank_scores(sentences: list[str], damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """PageRank scores of sentences on a word-overlap similarity graph.

    sim(Si, Sj) = |Si ∩ Sj| / (log(1 + |Si|) + log(1 + |Sj|))   (stop-words removed)
    PR = (1 - d) / N + d * P^T PR,  P = row-normalised similarity matrix
    """
    n = len(sentences)
    if n == 0:
        return np.zeros(0)
    token_sets = [set(t for t in word_tokenize(s) if t not in STOPWORDS) for s in sentences]
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, toks in enumerate(token_sets):
        for t in toks:
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
    if not vocab:
        return np.full(n, 1.0 / n)
    b = np.zeros((n, len(vocab)), dtype=np.float32)
    b[rows, cols] = 1.0
    overlap = b @ b.T                                   # |Si ∩ Sj|
    log_len = np.log1p(b.sum(axis=1))
    denom = log_len[:, None] + log_len[None, :]
    w = np.divide(overlap, denom, out=np.zeros_like(overlap), where=denom > 0)
    np.fill_diagonal(w, 0.0)
    out_weight = w.sum(axis=1)
    p = np.full((n, n), 1.0 / n, dtype=np.float64)       # dangling rows -> uniform jump
    has_out = out_weight > 0
    p[has_out] = w[has_out] / out_weight[has_out, None]
    scores = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = (1.0 - damping) / n + damping * (p.T @ scores)
        converged = np.abs(new - scores).sum() < tol
        scores = new
        if converged:
            break
    return scores


def textrank(text: str, k: int = 3, min_words: int = 4, **kwargs) -> str:
    """Top-k TextRank sentences, returned in document order."""
    sentences = split_sentences(text)
    idx = _candidate_indices(sentences, min_words)
    if len(idx) <= k:
        return "\n".join(sentences[i] for i in idx)
    scores = textrank_scores([sentences[i] for i in idx], **kwargs)
    top = sorted(range(len(idx)), key=lambda j: (-scores[j], j))[:k]   # ties -> earlier sentence
    return "\n".join(sentences[idx[j]] for j in sorted(top))


def mmr_select(embeddings: np.ndarray, k: int, lambda_: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance on L2-normalised embeddings, relevance = cosine to centroid."""
    n = len(embeddings)
    if n == 0:
        return []
    centroid = embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    relevance = embeddings @ centroid
    similarity = embeddings @ embeddings.T
    selected = [int(np.argmax(relevance))]
    while len(selected) < min(k, n):
        redundancy = similarity[:, selected].max(axis=1)
        score = lambda_ * relevance - (1.0 - lambda_) * redundancy
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))
    return selected


class EmbeddingExtractor:
    """Unsupervised extractive summarizer on top of a transformer sentence encoder.

    Default encoder: sentence-transformers/all-MiniLM-L6-v2 (22M params), mean pooling.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None,
                 batch_size: int = 256, max_length: int = 128, fp16: bool | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        if fp16 is None:
            fp16 = self.device.startswith("cuda")
        if fp16:
            self.model.half()
        self.model.to(self.device)
        self.batch_size, self.max_length = batch_size, max_length

    def encode(self, sentences: list[str]) -> np.ndarray:
        torch = self._torch
        order = np.argsort([-len(s) for s in sentences], kind="stable")   # length-sorted -> less padding
        dim = self.model.config.hidden_size
        out = np.zeros((len(sentences), dim), dtype=np.float32)
        for start in range(0, len(sentences), self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            enc = self.tokenizer([sentences[i] for i in batch_idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            out[batch_idx] = pooled.cpu().numpy()
        return out

    def summarize_many(self, texts: list[str], k: int = 3, lambda_: float = 0.7, min_words: int = 4,
                       docs_per_chunk: int = 2000, verbose: bool = False) -> list[str]:
        results: list[str] = []
        for c0 in range(0, len(texts), docs_per_chunk):
            docs, flat = [], []
            for text in texts[c0:c0 + docs_per_chunk]:
                sentences = split_sentences(text)
                idx = _candidate_indices(sentences, min_words)
                docs.append((sentences, idx, len(flat)))
                flat.extend(sentences[i] for i in idx)
            emb = self.encode(flat) if flat else np.zeros((0, 1), dtype=np.float32)
            for sentences, idx, offset in docs:
                if len(idx) <= k:
                    results.append("\n".join(sentences[i] for i in idx))
                    continue
                chosen = mmr_select(emb[offset:offset + len(idx)], k, lambda_)
                results.append("\n".join(sentences[idx[j]] for j in sorted(chosen)))
            if verbose:
                print(f"  embedding-MMR: {min(c0 + docs_per_chunk, len(texts))}/{len(texts)} documents")
        return results

    def summarize(self, text: str, k: int = 3, lambda_: float = 0.7) -> str:
        return self.summarize_many([text], k=k, lambda_=lambda_)[0]
