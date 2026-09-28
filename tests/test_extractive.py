import numpy as np

from src.extractive import lead_k, mmr_select, textrank, textrank_scores

DOC = ("The city council approved a new electric bus fleet on Monday. "
       "The electric buses will replace diesel buses on twelve routes. "
       "Officials said the bus fleet would cut emissions and fuel costs. "
       "Ok. "
       "A local bakery also celebrated its tenth anniversary with free bread. "
       "The council expects the first electric buses to arrive next spring.")


def test_lead_k_skips_short_fragments():
    out = lead_k(DOC, k=3).split("\n")
    assert len(out) == 3 and "Ok." not in out
    assert out[0].startswith("The city council")


def test_textrank_scores_are_a_distribution():
    from src.preprocessing import split_sentences
    scores = textrank_scores(split_sentences(DOC))
    assert np.isclose(scores.sum(), 1.0, atol=1e-6) and (scores > 0).all()


def test_textrank_prefers_central_sentences_and_keeps_order():
    out = textrank(DOC, k=2).split("\n")
    assert len(out) == 2
    assert all("bakery" not in s for s in out)                      # off-topic sentence is not central
    sentences = [s for s in DOC.split(". ")]
    positions = [DOC.index(s[:30]) for s in out]
    assert positions == sorted(positions)                            # document order preserved


def test_textrank_short_and_empty_documents():
    assert textrank("Only one sentence here in this text.", k=3) == "Only one sentence here in this text."
    assert textrank("", k=3) == ""
    assert textrank_scores([]).size == 0


def test_textrank_all_stopwords_uniform():
    scores = textrank_scores(["the and of", "is it was"])
    assert np.allclose(scores, 0.5)


def test_mmr_select_diversity():
    e = np.array([[1, 0], [0.99, 0.141], [0, 1], [0.7, 0.714]], dtype=np.float32)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    sel = mmr_select(e, k=2, lambda_=0.5)
    assert len(sel) == 2 and len(set(sel)) == 2
    assert not ({0, 1} <= set(sel))                                  # near-duplicates are not both chosen
    assert mmr_select(np.zeros((0, 2)), 3) == []
    assert len(mmr_select(e[:1], 3)) == 1
