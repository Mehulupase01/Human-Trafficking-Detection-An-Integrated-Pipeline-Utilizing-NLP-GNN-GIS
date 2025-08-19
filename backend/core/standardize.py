# backend/core/standardize.py
"""
Standardization & dedup helpers used across the pipeline.

This module intentionally has **no** heavy external dependencies.
If `rapidfuzz` is available, we use it to collapse near-duplicates.
Otherwise we fall back to case-insensitive exact matching.
"""

from __future__ import annotations
import re
from typing import Iterable, List, Tuple

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover
    _HAS_RAPIDFUZZ = False


_WHITESPACE_RE = re.compile(r'\s+')
# split on common list separators: comma, semicolon, slash, ampersand, ' and ', pipes
_LIST_SPLIT_RE = re.compile(r'\s*(?:,|;|/|&|\band\b|\||\u2013|\u2014|\u2192|->|=>| to )\s*', flags=re.IGNORECASE)


def clean_text(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip()
    s = _WHITESPACE_RE.sub(' ', s)
    # remove surrounding quotes
    s = s.strip('"\''"")
    return s


def split_list(value: str) -> List[str]:
    """
    Split a string with multiple possible separators into a list of tokens.
    """
    if value is None:
        return []
    value = clean_text(value)
    if not value:
        return []
    parts = _LIST_SPLIT_RE.split(value)
    # filter empty
    parts = [p.strip() for p in parts if p and p.strip() and p.strip().lower() not in {'nan', 'none', 'null', 'no'}]
    return parts


def standardize_gender(value: str) -> str:
    if not value:
        return ''
    v = clean_text(value).lower()
    mapping = {
        'm': 'Male', 'male': 'Male', 'man': 'Male', 'boy': 'Male',
        'f': 'Female', 'female': 'Female', 'woman': 'Female', 'girl': 'Female',
    }
    return mapping.get(v, value.title())


def titlecase_keep_caps(value: str) -> str:
    """
    Title-case while keeping common all-caps tokens like acronyms untouched.
    """
    if not value:
        return ''
    # Keep words like 'UK', 'USA' as-is
    words = clean_text(value).split(' ')
    out = []
    for w in words:
        if len(w) <= 4 and w.isupper():
            out.append(w)
        else:
            out.append(w.capitalize())
    return ' '.join(out).strip()


def standardize_nationality(value: str) -> str:
    if not value:
        return ''
    v = clean_text(value)
    # Basic normalization — users can extend this map as needed
    simple_map = {
        'ethiopian': 'Ethiopian',
        'eritrean': 'Eritrean',
        'sudanese': 'Sudanese',
        'somali': 'Somali',
        'nigerian': 'Nigerian',
        'libyan': 'Libyan',
        'syrian': 'Syrian',
        'afghan': 'Afghan',
        'afghani': 'Afghan',
    }
    key = v.lower()
    return simple_map.get(key, titlecase_keep_caps(v))


def standardize_location(value: str) -> str:
    if not value:
        return ''
    v = clean_text(value)
    # remove trailing dots / double spaces
    v = v.strip(' .')
    return titlecase_keep_caps(v)


def dedupe_preserve_order(items: Iterable[str], sim_threshold: int = 90) -> List[str]:
    """
    Deduplicate a list of strings while preserving the first-seen order.
    If rapidfuzz is available, collapse near-duplicates ≥ sim_threshold.
    """
    seen: List[str] = []
    for raw in items or []:
        s = clean_text(raw)
        if not s:
            continue
        # exact (case-insensitive) check first
        low = s.lower()
        if any(low == x.lower() for x in seen):
            continue
        if _HAS_RAPIDFUZZ:
            similar = False
            for prev in seen:
                if fuzz.ratio(prev.lower(), low) >= sim_threshold:
                    similar = True
                    break
            if similar:
                continue
        seen.append(s)
    return seen


def split_and_standardize_people(value: str) -> List[str]:
    tokens = split_list(value)
    tokens = [titlecase_keep_caps(t) for t in tokens]
    return dedupe_preserve_order(tokens)


def split_and_standardize_locations(value: str) -> List[str]:
    tokens = split_list(value)
    tokens = [standardize_location(t) for t in tokens]
    return dedupe_preserve_order(tokens)


def smart_blank(value):
    """
    Normalize pandas-friendly blanks: return '' for None/NaN-like strings.
    """
    if value is None:
        return ''
    v = str(value).strip()
    if v.lower() in {'nan', 'none', 'null'}:
        return ''
    return v
