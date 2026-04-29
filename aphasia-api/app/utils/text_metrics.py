"""Text-level error metrics for speech evaluation.

Provides Character Error Rate (CER) and Word Error Rate (WER) calculations
using edit-distance algorithms.  These metrics are essential for tracking
aphasia patient rehabilitation progress over time.
"""

from __future__ import annotations


def levenshtein_distance(source: str, target: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings.

    Uses the classic dynamic-programming approach with O(min(m, n)) space.
    """
    if len(source) < len(target):
        return levenshtein_distance(target, source)

    if len(target) == 0:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, s_char in enumerate(source):
        current_row = [i + 1]
        for j, t_char in enumerate(target):
            # Cost: 0 if characters match, 1 otherwise
            cost = 0 if s_char == t_char else 1
            current_row.append(
                min(
                    current_row[j] + 1,        # insertion
                    previous_row[j + 1] + 1,    # deletion
                    previous_row[j] + cost,     # substitution
                )
            )
        previous_row = current_row

    return previous_row[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER).

    CER = levenshtein(ref_chars, hyp_chars) / len(ref_chars)

    Returns:
        Float in [0.0, ∞).  0.0 means perfect match.
        Values > 1.0 are possible when the hypothesis is much longer
        than the reference.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "")

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / max(len(ref_chars), 1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER).

    WER = levenshtein(ref_words, hyp_words) / len(ref_words)

    Uses word-level Levenshtein distance.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = _word_levenshtein(ref_words, hyp_words)
    return dist / max(len(ref_words), 1)


def _word_levenshtein(source: list[str], target: list[str]) -> int:
    """Levenshtein distance operating on word tokens instead of characters."""
    if len(source) < len(target):
        return _word_levenshtein(target, source)

    if len(target) == 0:
        return len(source)

    previous_row = list(range(len(target) + 1))
    for i, s_word in enumerate(source):
        current_row = [i + 1]
        for j, t_word in enumerate(target):
            cost = 0 if s_word == t_word else 1
            current_row.append(
                min(
                    current_row[j] + 1,
                    previous_row[j + 1] + 1,
                    previous_row[j] + cost,
                )
            )
        previous_row = current_row

    return previous_row[-1]
