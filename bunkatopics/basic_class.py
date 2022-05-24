import pandas as pd
from .multiprocess_embeddings import get_embeddings
from .extract_terms import extract_terms_df
from .sent_multiprocessing import sent_multiprocessing, sent_extract
from .indexer import indexer
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BasicSemantics:
    """This Class carries out the basic operations that all following models will use
    - terms extraction
    - docs embeddings
    - terms embeddings
    - Sentence Extraction
    - terms indexation

    This class will be used as Parent of other specialized class

    """

    def __init__(
        self,
        data,
        text_var,
        index_var,
        terms_path=None,
        terms_embeddings_path=None,
        docs_embeddings_path=None,
    ):
        self.data = data[data[text_var].notna()].reset_index(drop=True)
        self.text_var = text_var
        self.index_var = index_var

        self.terms_path = terms_path
        self.terms_embeddings_path = terms_embeddings_path
        self.docs_embeddings_path = docs_embeddings_path

        # Load existing dataset if they exist
        """if terms_path is not None:
            self.terms = pd.read_csv(terms_path, index_col=[0])
            self.index_terms(projection=False, db_path=".")"""

        if terms_embeddings_path is not None:
            self.terms_embeddings = pd.read_csv(terms_embeddings_path, index_col=[0])

        if docs_embeddings_path is not None:
            self.docs_embeddings = pd.read_csv(docs_embeddings_path, index_col=[0])

    def fit(
        self,
        extract_terms=True,
        terms_embeddings=True,
        docs_embeddings=False,
        sample_size_terms=500,
        terms_limit=500,
        terms_ents=True,
        terms_ngrams=(1, 2),
        terms_ncs=True,
        terms_include_pos=["NOUN", "PROPN", "ADJ"],
        terms_include_types=["PERSON", "ORG"],
        embeddings_model="distiluse-base-multilingual-cased-v1",
        multiprocessing=True,
        reduction=5,
        language="en",
    ):

        if self.terms_embeddings_path is not None:
            terms_embeddings = False
        if self.docs_embeddings_path is not None:
            docs_embeddings = False
        if self.terms_path is not None:
            extract_terms = False

        if extract_terms:
            logging.info("Extracting Terms...")
            self.terms = self.extract_terms(
                sample_size=sample_size_terms,
                limit=terms_limit,
                ents=terms_ents,
                ncs=terms_ncs,
                ngrams=terms_ngrams,
                include_pos=terms_include_pos,
                include_types=terms_include_types,
                language=language,
                multiprocessing=multiprocessing,
            )

        if docs_embeddings:
            logging.info("Extracting Docs Embeddings...")
            self.docs_embeddings = self.extract_docs_embeddings(
                multiprocessing=multiprocessing, reduction=reduction
            )
        if terms_embeddings:
            logging.info("Extracting Terms Embeddings...")
            self.terms_embeddings = self.extract_terms_embeddings(
                multiprocessing=multiprocessing, reduction=reduction
            )

        self.docs_embedding_model = embeddings_model

        self.data = self.data.set_index(self.index_var)

    def extract_terms(
        self,
        sample_size,
        limit,
        ents=True,
        ncs=True,
        ngrams=(1, 2),
        include_pos=["NOUN", "PROPN", "ADJ"],
        include_types=["PERSON", "ORG"],
        language="en",
        multiprocessing=True,
    ):
        self.terms, terms_indexed = extract_terms_df(
            self.data,
            text_var=self.text_var,
            index_var=self.index_var,
            ngs=True,
            ents=ents,
            ncs=ncs,
            multiprocess=multiprocessing,
            sample_size=sample_size,
            drop_emoji=True,
            ngrams=ngrams,
            remove_punctuation=False,
            include_pos=include_pos,
            include_types=include_types,
            language=language,
        )

        self.terms = self.terms.sort_values("count_terms", ascending=False)
        self.terms = self.terms.head(limit)

        self.df_terms_indexed = terms_indexed.copy()

        return self.terms

    def extract_docs_embeddings(self, multiprocessing=True, reduction=5):
        # Extract Embeddings
        self.df_docs_embeddings = get_embeddings(
            self.data,
            index_var=self.index_var,
            text_var=self.text_var,
            multiprocessing=multiprocessing,
            reduction=reduction,
        )

        return self.df_docs_embeddings

    def extract_terms_embeddings(self, multiprocessing=True, reduction=5):
        # Extract Embeddings
        self.df_terms_embeddings = get_embeddings(
            self.terms.reset_index(),
            index_var="index",
            text_var="text",
            multiprocessing=multiprocessing,
            reduction=reduction,
        )

        self.terms = self.terms.drop("index", axis=1)

        self.df_terms_embeddings.index = self.terms.index

        return self.df_terms_embeddings

    def extract_sentences(self, multiprocess=True):
        docs = self.data[self.text_var]
        if multiprocess:
            res = sent_multiprocessing(docs)
        else:
            res = sent_extract(docs)

        indexes = self.data[self.index_var].to_list()
        final_dict = {"sentences": res, "doc_index": indexes}

        self.df_sent = pd.DataFrame(final_dict)

        return self.df_sent

    def index_terms(self, list_to_index):

        docs = self.data[self.text_var].to_list()
        res = indexer(list_to_index, docs)

        res = res.groupby("docs")["indexed_term"].apply(list).reset_index()
        res.columns = [self.text_var, "text"]
        res = pd.merge(
            res, self.data[[self.text_var, self.index_var]], on=self.text_var
        )

        final_res = res.copy()
        self.df_terms_indexed = final_res.drop(self.text_var, axis=1)
        self.df_terms_indexed = self.df_terms_indexed.set_index(self.index_var)

        return res
