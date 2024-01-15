import typing as t


def remove_overlapping_terms(terms: t.List[str]) -> t.List[str]:
    """
    Remove overlapping terms from a list of terms.

    Args:
        terms (List[str]): List of terms to process.

    Returns:
        List[str]: List of terms with overlapping terms removed.
    """
    seen_words = set()
    filtered_terms = []

    for term in terms:
        term_words = term.split()
        if not any(word.lower() in seen_words for word in term_words):
            filtered_terms.append(term)
            seen_words.update(word.lower() for word in term_words)

    return filtered_terms
