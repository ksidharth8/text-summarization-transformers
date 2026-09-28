import math

import numpy as np

from src.metrics import (bootstrap_ci, evaluate_system, length_stats, novel_ngram_ratio, paired_bootstrap,
                         rouge_per_sample)


def test_rouge_identity_and_disjoint():
    s = rouge_per_sample(["the cat sat on the mat.", "alpha beta"], ["the cat sat on the mat.", "gamma delta"])
    for t in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
        assert math.isclose(s[t][0], 100.0) and s[t][1] == 0.0


def test_rouge_known_value():
    # unigram overlap: pred 4 tokens, ref 6 tokens, 4 common -> P=1, R=2/3, F=0.8
    s = rouge_per_sample(["police arrested two men"], ["police arrested two men on friday"])
    assert math.isclose(s["rouge1"][0], 80.0, rel_tol=1e-9)


def test_rouge_lsum_uses_sentence_split():
    pred = "Police arrested two men. The fire spread quickly across town."
    ref = "The fire spread quickly across town. Police arrested two men."
    s = rouge_per_sample([pred], [ref])
    assert s["rougeLsum"][0] > s["rougeL"][0]                        # summary-level LCS is order-robust


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(1).normal(40, 10, 500)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi and hi - lo < 4
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    b = rng.normal(40, 5, 400)
    r = paired_bootstrap(b + 2.0, b)
    assert math.isclose(r["diff"], 2.0, abs_tol=1e-6) and r["p_value"] == 0.0
    assert math.isnan(paired_bootstrap([1, 2], [1])["diff"])


def test_novel_ngrams():
    r = novel_ngram_ratio(["the cat sat on the mat"], ["the cat sat on the sofa"])
    assert r["1"] == round(100 / 5, 2)                                # 'sofa' is the only new unigram of 5 unique
    assert r["4"] > r["1"]
    assert novel_ngram_ratio(["x"], [""])["1"] != novel_ngram_ratio(["x"], [""])["1"]   # nan when empty


def test_length_stats_and_evaluate_system():
    assert length_stats(["a b c.", "d e."]) == {"avg_words": 2.5, "median_words": 2.5, "avg_sentences": 1.0}
    res, per = evaluate_system(["the cat sat.", "dogs bark loudly."], ["the cat sat.", "dogs bark."],
                               sources=["the cat sat on a mat.", "dogs bark."], novelty_idx=[0, 1])
    assert res["n"] == 2 and "rouge1_ci" in res and res["novel_ngrams"]["1"] >= 0
    assert per["rouge1"].shape == (2,)
