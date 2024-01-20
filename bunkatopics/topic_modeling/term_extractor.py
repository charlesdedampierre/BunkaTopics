import typing as t
import warnings
from functools import partial

import pandas as pd
import spacy
import textacy
import textacy.preprocessing
from tqdm import tqdm

from bunkatopics.datamodel import Term

# Define a preprocessing pipeline
preproc = textacy.preprocessing.make_pipeline(
    textacy.preprocessing.normalize.unicode,
    textacy.preprocessing.normalize.bullet_points,
    textacy.preprocessing.normalize.quotation_marks,
    textacy.preprocessing.normalize.whitespace,
    textacy.preprocessing.normalize.hyphenated_words,
    textacy.preprocessing.remove.brackets,
    textacy.preprocessing.replace.currency_symbols,
    textacy.preprocessing.remove.html_tags,
)

# Define custom types for document and term IDs
DOC_ID = t.TypeVar("DOC_ID")
TERM_ID = t.TypeVar("TERM_ID")


# Suppress specific category of warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class TextacyTermsExtractor:

    """
    Extracts terms from text using Textacy and SpaCy libraries.

    This class provides functionalities to extract terms from a given list of documents,
    considering various linguistic features like n-grams, named entities, and noun chunks.

    """

    def __init__(
        self,
        language: str = "english",
        ngrams: t.List[int] = [1, 2, 3],
        ngs: bool = True,
        ents: bool = False,
        ncs: bool = False,
        drop_emoji: bool = True,
        include_pos: t.List[str] = ["NOUN"],
        include_types: t.List[str] = ["PERSON", "ORG"],
    ):
        """
        Initializes the TextacyTermsExtractor with specified configuration.

        Args:
            language (str): The language of the text. Defaults to 'english'.
            ngrams (tuple[int, ...]): Tuple of n-gram lengths to consider. Defaults to (1, 2, 3).
            ngs (bool): Include n-grams in extraction. Defaults to True.
            ents (bool): Include named entities in extraction. Defaults to True.
            ncs (bool): Include noun chunks in extraction. Defaults to True.
            drop_emoji (bool): Remove emojis before extraction. Defaults to True.
            include_pos (list[str]): POS tags to include. Defaults to ["NOUN"].
            include_types (list[str]): Entity types to include. Defaults to ["PERSON", "ORG"].

        Raises:
            ValueError: If the specified language is not supported.
        """

        # Supported languages with corresponding SpaCy models
        self.supported_languages = {
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

        self.ngs = ngs
        self.ents = ents
        self.ncs = ncs
        self.drop_emoji = drop_emoji
        self.include_pos = include_pos
        self.include_types = include_types
        self.ngrams = ngrams

        # Set the language model based on the provided language
        if language in self.supported_languages:
            self.language_model = self.supported_languages[language]
        else:
            raise ValueError(f"The language '{language}' is not supported.")

        try:
            spacy.load(self.language_model)
        except OSError:
            # The model is not installed, so download it
            spacy.cli.download(self.language_model)

    def fit_transform(
        self,
        ids: t.List[DOC_ID],
        sentences: t.List[str],
    ) -> t.Tuple[t.List[Term], t.Dict[DOC_ID, t.List[TERM_ID]]]:
        """
        Extracts terms from the provided documents and returns them along with their indices.

        Args:
            ids (List[DOC_ID]): List of document IDs.
            sentences (List[str]): List of sentences corresponding to the document IDs.

        Notes:
            - The method processes each document to extract relevant terms based on the configured
            linguistic features such as n-grams, named entities, and noun chunks.
            - It also handles pre-processing steps like normalizing text, removing brackets,
            replacing currency symbols, removing HTML tags, and optionally dropping emojis.
        """

        # Create a DataFrame from the provided document IDs and sentences
        self.df = pd.DataFrame({"content": sentences, "doc_id": ids})

        # Extract terms from the DataFrame
        df_terms, df_terms_indexed = self.extract_terms_df(
            self.df,
            text_var="content",
            index_var="doc_id",
            ngs=self.ngs,
            ents=self.ents,
            ncs=self.ncs,
            sample_size=100000,
            drop_emoji=self.drop_emoji,
            ngrams=self.ngrams,
            remove_punctuation=True,
            include_pos=self.include_pos,
            include_types=self.include_types,
            language_model=self.language_model,
        )

        # Process and return the extracted terms
        df_terms = df_terms.reset_index().rename(columns={"terms_indexed": "term_id"})
        terms = [Term(**row) for row in df_terms.to_dict(orient="records")]
        self.terms: t.List[Term] = terms

        df_terms_indexed = df_terms_indexed.reset_index().rename(
            columns={"text": "terms_indexed"}
        )
        indexed_terms_dict = df_terms_indexed.set_index("doc_id")[
            "terms_indexed"
        ].to_dict()

        return terms, indexed_terms_dict

    def extract_terms_df(
        self,
        data: pd.DataFrame,
        text_var: str,
        index_var: str,
        ngs: bool = True,
        ents: bool = True,
        ncs: bool = False,
        sample_size: int = 100000,
        drop_emoji: bool = True,
        ngrams: t.Tuple[int, int] = (2, 2),
        remove_punctuation: bool = False,
        include_pos: t.List[str] = ["NOUN", "PROPN", "ADJ"],
        include_types: t.List[str] = ["PERSON", "ORG"],
        language_model: str = "en_core_web_sm",
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        load_lang = textacy.load_spacy_lang(language_model, disable=())

        def extract_terms(
            tuple: t.Tuple[int, str],
            ngs: bool,
            ents: bool,
            ncs: bool,
            ngrams: t.Tuple[int, int],
            drop_emoji: bool,
            remove_punctuation: bool,
            include_pos: t.List[str],
            include_types: t.List[str],
        ) -> pd.DataFrame:
            index = tuple[0]
            text = tuple[1]

            prepro_text = preproc(str(text))
            if drop_emoji:
                prepro_text = textacy.preprocessing.replace.emojis(prepro_text, repl="")

            if remove_punctuation:
                prepro_text = textacy.preprocessing.remove.punctuation(prepro_text)

            doc = textacy.make_spacy_doc(prepro_text, lang=load_lang)

            terms = []

            if ngs:
                ngrams_terms = list(
                    textacy.extract.terms(
                        doc,
                        ngs=partial(
                            textacy.extract.ngrams,
                            n=ngrams,
                            filter_punct=True,
                            filter_stops=True,
                            include_pos=include_pos,
                        ),
                        dedupe=False,
                    )
                )

                terms.append(ngrams_terms)

            if ents:
                ents_terms = list(
                    textacy.extract.terms(
                        doc,
                        ents=partial(
                            textacy.extract.entities, include_types=include_types
                        ),
                        dedupe=False,
                    )
                )
                terms.append(ents_terms)

            if ncs:
                ncs_terms = list(
                    textacy.extract.terms(
                        doc,
                        ncs=partial(textacy.extract.noun_chunks, drop_determiners=True),
                        dedupe=False,
                    )
                )

                noun_chunks = [x for x in ncs_terms if len(x) >= 3]
                terms.append(noun_chunks)

            final = [item for sublist in terms for item in sublist]
            final = list(set(final))

            df = [
                (term.text, term.lemma_.lower(), term.label_, term.__len__())
                for term in final
            ]
            df = pd.DataFrame(df, columns=["text", "lemma", "ent", "ngrams"])
            df["text_index"] = index

            return df

        data = data[data[text_var].notna()]
        data = data.sample(min(sample_size, len(data)))

        sentences = data[text_var].tolist()
        indexes = data[index_var].tolist()
        inputs = [(x, y) for x, y in zip(indexes, sentences)]

        res = list(
            tqdm(
                map(
                    partial(
                        extract_terms,
                        ngs=ngs,
                        ents=ents,
                        ncs=ncs,
                        drop_emoji=drop_emoji,
                        remove_punctuation=remove_punctuation,
                        ngrams=ngrams,
                        include_pos=include_pos,
                        include_types=include_types,
                    ),
                    inputs,
                ),
                total=len(inputs),
            )
        )

        final_res = pd.concat([x for x in res])

        terms = (
            final_res.groupby(["text", "lemma", "ent", "ngrams"])
            .agg(count_terms=("text_index", "count"))
            .reset_index()
        )

        terms = terms.sort_values(["text", "ent"]).reset_index(drop=True)
        terms = terms.drop_duplicates(["text"], keep="first")
        terms = terms.sort_values("count_terms", ascending=False)
        terms = terms.rename(columns={"text": "terms_indexed"})
        terms = terms.set_index("terms_indexed")

        terms_indexed = final_res[["text", "text_index"]].drop_duplicates()
        terms_indexed = terms_indexed.rename(columns={"text_index": index_var})
        terms_indexed = terms_indexed.groupby(index_var)["text"].apply(list)
        terms_indexed = terms_indexed.reset_index()
        terms_indexed = terms_indexed.rename(columns={"text": "terms_indexed"})
        terms_indexed = terms_indexed.set_index(index_var)

        return terms, terms_indexed


def from_dict_to_frame(indexed_dict):
    data = {k: [v] for k, v in indexed_dict.items()}
    df = pd.DataFrame.from_dict(data).T
    df.columns = ["text"]
    df = df.explode("text")
    return df
