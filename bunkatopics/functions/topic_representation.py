import typing as t


def remove_overlapping_terms(terms: t.List[str]) -> t.List[str]:
    # 'Harry Style | Harry | Style -> Harry Style'
    seen_words = set()
    filtered_terms = []
    for term in terms:
        if not any(word.lower() in seen_words for word in term.split()):
            filtered_terms.append(term)
            seen_words.update([w.lower() for w in term.split()])
    return filtered_terms
