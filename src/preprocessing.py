"""Text cleaning and sentence splitting (pure Python, no downloads needed).

CNN/DailyMail articles carry scraping boilerplate that hurts extractive baselines and wastes
encoder tokens, e.g.

    "By . Associated Press . PUBLISHED: . 14:11 EST, 25 October 2013 . | . UPDATED: . ..."
    "WASHINGTON (CNN) -- ..."

`clean_article` removes these conservatively (only well-defined patterns). Reference summaries
are never changed in content: `clean_summary` only normalises whitespace and keeps one
highlight per line, which is what ROUGE-Lsum expects.
"""
from __future__ import annotations

import re

__all__ = ["clean_article", "clean_summary", "split_sentences", "word_tokenize", "count_words"]

# DailyMail time stamps: "PUBLISHED: . 14:11 EST, 25 October 2013 . | . "
_TIMESTAMP = re.compile(
    r"(?:PUBLISHED|UPDATED)\s*:\s*\.?\s*\d{1,2}:\d{2}\s*(?:[AaPp]\.?[Mm]\.?\s*)?(?:[A-Z]{2,4}\s*)?,?\s*"
    r"\d{1,2}\s+[A-Za-z]+\s+\d{4}\s*\.?\s*(?:\|\s*\.?\s*)?"
)
# "Last updated at 3:23 PM on 12th January 2012 ."
_LAST_UPDATED = re.compile(
    r"Last updated (?:at|on)\s*\d{1,2}:\d{2}\s*(?:[AaPp]\.?[Mm]\.?)?\s*(?:on\s*)?"
    r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\s*\.?\s*"
)
# "By . Daily Mail Reporter . " / "By . Jane Doe . and John Roe . " (only at the very start)
_BYLINE = re.compile(r"^By\s*\.\s*[^.]{1,100}?\s*\.\s*(?:(?:and|&)\s*[^.]{1,100}?\s*\.\s*)?")
# "(CNN) -- ", "LONDON, England (CNN) -- ", "(CNN)Hillary ..." (only at the very start)
_DATELINE = re.compile(   # place name = up to 6 capitalised words ("LONDON, England", "Hong Kong")
    r"^(?:[A-Z][A-Za-z.'-]*(?:,?\s+[A-Z][A-Za-z.'-]*){0,5},?\s*)?"
    r"\((?:CNN|CNN Student News|EW\.com|Mashable|Reuters|AP)\)\s*(?:--|-|\u2014|\u2013)?\s*"
)
_SCROLL = re.compile(r"\bScroll down for (?:video|videos)\s*\.?\s*", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.!?,;:])(?=\s|$)")
_WS = re.compile(r"[ \t\r\f\v]+")
_ANY_WS = re.compile(r"\s+")

# Tokens that end with '.' but usually do not end a sentence.
_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st mt vs etc inc ltd co corp jan feb mar apr jun jul aug sep sept oct
    nov dec gen gov sen rep col lt sgt capt cmdr adm rev fig figs eq eqs al approx dept univ ave blvd
    no nos vol pp ch sec u.s u.k u.n e.g i.e a.m p.m ph.d""".split()
)
_BOUNDARY = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s+[\"'\u201c\u2018(\[]?[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9]+")


def clean_article(text: str) -> str:
    """Remove CNN/DailyMail scraping boilerplate and normalise whitespace (single line output)."""
    if not isinstance(text, str):
        return ""
    t = _ANY_WS.sub(" ", text).strip()
    head, tail = t[:600], t[600:]            # boilerplate only ever appears at the top
    head = _TIMESTAMP.sub("", head)
    head = _LAST_UPDATED.sub("", head)
    head = _BYLINE.sub("", head.lstrip())
    head = _DATELINE.sub("", head.lstrip())
    t = head + tail
    t = _SCROLL.sub("", t)
    t = _SPACE_BEFORE_PUNCT.sub(r"\1", t)
    return _ANY_WS.sub(" ", t).strip()


def clean_summary(text: str) -> str:
    """Normalise a reference summary: one sentence/highlight per line, tidy spacing."""
    if not isinstance(text, str):
        return ""
    lines = []
    for line in text.replace("\r", "\n").split("\n"):
        line = _WS.sub(" ", line).strip()
        line = _SPACE_BEFORE_PUNCT.sub(r"\1", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def split_sentences(text: str) -> list[str]:
    """Abbreviation-aware rule-based sentence splitter. Newlines are always sentence boundaries."""
    if not isinstance(text, str) or not text.strip():
        return []
    sentences: list[str] = []
    for line in re.split(r"\s*\n+\s*", text.strip()):
        line = _ANY_WS.sub(" ", line).strip()
        if not line:
            continue
        start = 0
        for m in _BOUNDARY.finditer(line):
            prev_token = line[start:m.start()].rsplit(" ", 1)[-1].lower().strip("\"'(\u201c\u2018")
            if (
                prev_token in _ABBREVIATIONS
                or re.fullmatch(r"[a-z]", prev_token)                 # initials: "J. K. Rowling"
                or re.fullmatch(r"(?:[a-z]\.)+[a-z]", prev_token)     # "U.S", "e.g"
            ):
                continue
            sent = line[start:m.end()].strip()
            if sent:
                sentences.append(sent)
            start = m.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def word_tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens (same normalisation as the rouge-score tokenizer)."""
    return _WORD.findall(text.lower()) if isinstance(text, str) else []


def count_words(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0
