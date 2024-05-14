from collections import Counter

import numpy as np
import pandas as pd
from langdetect import LangDetectException, detect
from bunkatopics.logging import logger


def specificity(
    df: pd.DataFrame, X: str, Y: str, Z: str, top_n: int = 50
) -> pd.DataFrame:
    """
    Calculate specificity between two categorical variables X and Y in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing data.
        X (str): The name of the first categorical variable.
        Y (str): The name of the second categorical variable.
        Z (str): The name of the column to be used as a weight (if None, default weight is 1).
        top_n (int): The number of top results to return.

    Returns:
        pd.DataFrame: A DataFrame containing specificity scores between X and Y.
    """
    if Z is None:
        Z = "count_values"
        df[Z] = 1
        group = df.groupby([X, Y]).agg(count_values=(Z, "sum")).reset_index()
        cont = group.pivot(index=X, columns=Y, values=Z).fillna(0).copy()
    else:
        group = df.groupby([X, Y])[Z].sum().reset_index()
        cont = group.pivot(index=X, columns=Y, values=Z).fillna(0).copy()

    tx = df[X].value_counts()
    ty = df[Y].value_counts()

    cont = cont.astype(int)

    tx_df = pd.DataFrame(tx)
    tx_df.columns = ["c"]
    ty_df = pd.DataFrame(ty)
    ty_df.columns = ["c"]

    # Total observed values
    n = group[Z].sum()

    # Matrix product using pd.T to pivot one of the two series.
    indep = tx_df.dot(ty_df.T) / n

    cont = cont.reindex(indep.columns, axis=1)
    cont = cont.reindex(indep.index, axis=0)

    # Contingency Matrix
    ecart = (cont - indep) ** 2 / indep
    chi2 = ecart.sum(axis=1)
    chi2 = chi2.sort_values(ascending=False)
    spec = ecart * np.sign(cont - indep)

    # Edge Table of X, Y, specificity measure
    edge = spec.unstack().reset_index()
    edge = edge.rename(columns={0: "specificity_score"})
    edge = edge.sort_values(
        by=[X, "specificity_score"], ascending=[True, False]
    ).reset_index(drop=True)
    edge = edge[edge["specificity_score"] > 0]
    edge = edge.groupby([X]).head(top_n)
    edge = edge.reset_index(drop=True)

    return edge


def most_common_element(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None


def detect_language(documents):
    langs = []
    for doc in documents:
        try:
            lang = detect(doc)
            langs.append(lang)
        except:
            logger.debug(f"Could not detect language for document: {doc}")
            pass

    res = most_common_element(langs)

    return res


detect_language_to_spacy_model = {
    "ar": "ar_core_news_sm",  # Arabic
    "da": "da_core_news_sm",  # Danish
    "de": "de_core_news_sm",  # German
    "el": "el_core_news_sm",  # Greek
    "en": "en_core_web_sm",  # English
    "es": "es_core_news_sm",  # Spanish
    "fa": "fa_core_news_sm",  # Persian
    "fr": "fr_core_news_sm",  # French
    "it": "it_core_news_sm",  # Italian
    "ja": "ja_core_news_sm",  # Japanese
    "no": "nb_core_news_sm",  # Norwegian
    "pl": "pl_core_news_sm",  # Polish
    "pt": "pt_core_news_sm",  # Portuguese
    "ro": "ro_core_news_sm",  # Romanian
    "ru": "ru_core_news_sm",  # Russian
    "sv": "sv_core_news_sm",  # Swedish
    "tr": "tr_core_news_sm",  # Turkish
    "zh-cn": "zh_core_web_sm",  # Chinese Simplified
    "zh-tw": "zh_core_web_sm",  # Chinese Traditional
}

detect_language_to_language_name = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "tr": "Turkish",
    "zh-cn": "Chinese Simplified",
    "zh-tw": "Chinese Traditional",
}

# Supported languages with corresponding SpaCy models
old_supported_languages = {
    "english": "en_core_web_sm",
    "spanish": "es_core_news_sm",
    "french": "fr_core_news_sm",
    "german": "de_core_news_sm",
    "arabic": "ar_core_news_sm",
    "chinese": "zh_core_web_sm",
    "danish": "da_core_news_sm",
    "dutch": "nl_core_news_sm",
    "greek": "el_core_news_sm",
    "italian": "it_core_news_sm",
    "japanese": "ja_core_news_sm",
    "norwegian": "nb_core_news_sm",
    "polish": "pl_core_news_sm",
    "portuguese": "pt_core_news_sm",
    "romanian": "ro_core_news_sm",
    "russian": "ru_core_news_sm",
    "swedish": "sv_core_news_sm",
    "turkish": "tr_core_news_sm",
    "multilingual": "xx_ent_wiki_sm",  # Multilingual model
    # Additional languages can be added here
}
