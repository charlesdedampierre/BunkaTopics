import os
import random
import unittest

from FlagEmbedding import FlagModel
import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import ast
import umap

load_dotenv()
from langchain_community.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import HuggingFaceEmbeddings
from bunkatopics import Bunka

random.seed(42)


figure = True

dataset = pd.read_csv("big_data/lemonde_bunka_sample.csv")
docs = dataset["titles"].tolist()


class TestBunka(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        projection_model = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=3,
            random_state=42,
        )
        embedding_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
        cls.bunka = Bunka(
            projection_model=projection_model, embedding_model=embedding_model
        )
        # metadata = None
        cls.bunka.fit(
            docs=docs,
            pre_computed_embeddings=None,
            sampling_size_for_terms=1000,
        )

    def test_embed_sentence_transformer(self):

        projection_model = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=3,
            random_state=42,
        )

        embedding_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")

        bunka = Bunka(
            projection_model=projection_model, embedding_model=embedding_model
        )

        print("Fitting Bunka with SentenceTransformer")

        bunka.fit(
            docs=docs,
            pre_computed_embeddings=None,
            sampling_size_for_terms=1000,
        )


if __name__ == "__main__":
    unittest.main()
