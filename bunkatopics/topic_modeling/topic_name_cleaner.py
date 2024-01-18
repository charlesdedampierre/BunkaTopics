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
        # Remove leading and trailing spaces and convert to lowercase
        cleaned_term = term.strip()

        # Skip terms with one letter or number or with only alpha-numeric sign
        if (
            len(cleaned_term) <= 1
            or cleaned_term.isnumeric()
            or not cleaned_term.isalpha()
        ):
            continue

        # Check if the cleaned term consists of only alphabetical characters
        if all(char.isalpha() for char in cleaned_term):
            # Check if the cleaned term is in the seen_words set
            if cleaned_term not in seen_words:
                filtered_terms.append(cleaned_term)
                seen_words.add(cleaned_term)

    return filtered_terms
