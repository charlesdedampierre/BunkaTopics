from typing import List
import pandas as pd
import warnings

from numba.core.errors import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
import os
import random
import string
import typing as t
import uuid
import warnings
import subprocess
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import Chroma
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from bunkatopics.bunka_logger import logger
from bunkatopics.datamodel import (
    DOC_ID,
    TERM_ID,
    TOPIC_ID,
    BourdieuQuery,
    Document,
    Term,
    Topic,
    TopicGenParam,
    TopicParam,
)
from bunkatopics.functions.bourdieu_api import bourdieu_api
from bunkatopics.functions.coherence import get_coherence
from bunkatopics.functions.extract_terms import extract_terms_df
from bunkatopics.functions.topic_document import get_top_documents
from bunkatopics.functions.topic_gen_representation import get_clean_topic_all
from bunkatopics.functions.topic_utils import get_topic_repartition
from bunkatopics.functions.topics_modeling import get_topics
from bunkatopics.visualisation.bourdieu_visu import visualize_bourdieu_one_dimension
from bunkatopics.visualisation.new_bourdieu_visu import visualize_bourdieu
from bunkatopics.visualisation.query_visualisation import plot_query
from bunkatopics.visualisation.topic_visualization import visualize_topics
from bunkatopics.serveur.utils import is_server_running, kill_server


class Bunka:
    # ...

    def get_topics(
        self,
        n_clusters: int = 5,
        ngrams: List[int] = [1, 2],
        name_length: int = 15,
        top_terms_overall: int = 2000,
        min_count_terms: int = 2,
    ) -> pd.DataFrame:
        """
        Generate topics from the documents by clustering and assign topic names using a generative model.

        Args:
            n_clusters (int, optional): The number of clusters to generate. Default is 5.
            ngrams (List[int], optional): The ngrams to use for clustering. Default is [1, 2].
            name_length (int, optional): The maximum length of the generated topic names. Default is 15.
            top_terms_overall (int, optional): The number of top terms to use overall for clustering. Default is 2000.
            min_count_terms (int, optional): The minimum document count for terms to be included. Default is 2.

        Returns:
            pd.DataFrame: A DataFrame containing topic information including name, terms, and document counts.
        """
        self.topics: List[Topic] = get_topics(
            docs=self.docs,
            terms=self.terms,
            n_clusters=n_clusters,
            ngrams=ngrams,
            name_length=name_length,
            x_column="x",
            y_column="y",
            top_terms_overall=top_terms_overall,
            min_count_terms=min_count_terms,
        )

        self.docs, self.topics = get_top_documents(
            self.docs, self.topics, ranking_terms=20
        )
        df_topics = pd.DataFrame.from_records([topic.dict() for topic in self.topics])
        return df_topics
