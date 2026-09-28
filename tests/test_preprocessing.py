from src.preprocessing import clean_article, clean_summary, count_words, split_sentences, word_tokenize


def test_dailymail_boilerplate_removed():
    raw = ("By . Daily Mail Reporter . PUBLISHED: . 14:11 EST, 25 October 2013 . | . UPDATED: . "
           "15:20 EST, 25 October 2013 . A council in Leeds has approved a new cycle lane . "
           "Scroll down for video . Residents welcomed the decision .")
    out = clean_article(raw)
    assert out.startswith("A council in Leeds has approved a new cycle lane.")
    assert "PUBLISHED" not in out and "UPDATED" not in out and "Scroll down" not in out
    assert out.endswith("Residents welcomed the decision.")


def test_cnn_dateline_removed():
    assert clean_article("WASHINGTON (CNN) -- The Senate voted on Tuesday.") == "The Senate voted on Tuesday."
    assert clean_article("(CNN)The storm weakened overnight.") == "The storm weakened overnight."
    assert clean_article("LONDON, England (CNN) -- Rain fell.") == "Rain fell."
    assert clean_article("Hong Kong (CNN)Protests grew.") == "Protests grew."


def test_dateline_only_at_start():
    text = "Officials said the report, first published by (CNN) -- partners, was accurate."
    assert "(CNN)" in clean_article(text)


def test_non_string_inputs():
    assert clean_article(None) == "" and clean_summary(float("nan")) == ""
    assert split_sentences("") == [] and word_tokenize(None) == [] and count_words(None) == 0


def test_clean_summary_keeps_one_highlight_per_line():
    assert clean_summary("First point .\n\n  Second   point !\r\nThird") == "First point.\nSecond point!\nThird"


def test_split_sentences_abbreviations_and_initials():
    text = "Dr. Smith met J. K. Rowling in London. They spoke for 2.5 hours. Was it fun? Yes!"
    assert split_sentences(text) == ["Dr. Smith met J. K. Rowling in London.", "They spoke for 2.5 hours.",
                                      "Was it fun?", "Yes!"]


def test_split_sentences_quotes_and_newlines():
    text = 'He said "we will win." Then he left.\nSecond line without period'
    assert split_sentences(text) == ['He said "we will win."', "Then he left.", "Second line without period"]


def test_word_tokenize_matches_rouge_normalisation():
    assert word_tokenize("It's 3.5km, U.S.-based!") == ["it", "s", "3", "5km", "u", "s", "based"]
