from src.summarizer import DEFAULT_GENERATION, chunk_sentences, is_t5_like


def count(text):                      # 1 token per whitespace-separated word
    return len(text.split())


def test_chunks_respect_budget_and_cover_everything():
    sentences = [f"s{i} " + "w " * (i % 7 + 3) for i in range(40)]
    chunks = chunk_sentences(sentences, count, max_tokens=30, overlap=1)
    for c in chunks:
        assert sum(count(s) + 1 for s in c) <= 30
    covered = {s for c in chunks for s in c}
    assert covered == set(sentences)
    for a, b in zip(chunks, chunks[1:]):                              # one sentence of overlap
        assert a[-1] == b[0]


def test_no_overlap():
    sentences = ["a b c d"] * 10
    chunks = chunk_sentences(sentences, count, max_tokens=12, overlap=0)
    assert sum(len(c) for c in chunks) == 10


def test_overlong_sentence_is_split():
    long_sentence = " ".join(f"w{i}" for i in range(95))
    chunks = chunk_sentences(["short one.", long_sentence, "tail."], count, max_tokens=20, overlap=1)
    assert all(sum(count(s) + 1 for s in c) <= 20 for c in chunks)
    words = " ".join(" ".join(c) for c in chunks).split()
    assert all(f"w{i}" in words for i in range(95))


def test_single_small_document_one_chunk():
    assert chunk_sentences(["one two", "three"], count, max_tokens=50) == [["one two", "three"]]


def test_misc():
    assert is_t5_like("t5") and not is_t5_like("bart")
    assert DEFAULT_GENERATION["num_beams"] == 4 and DEFAULT_GENERATION["min_new_tokens"] < DEFAULT_GENERATION["max_new_tokens"]
