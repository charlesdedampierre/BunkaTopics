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

import torch


import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


device = torch.device("cpu")

random.seed(42)

# repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
# llm = HuggingFaceHub(
#     repo_id=repo_id,
#     huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
# )

figure = True


# Preprocess a dataset
dataset = load_dataset("bunkalab/medium-sample-technology")
df_test = pd.DataFrame(dataset["train"])

df_test = df_test[["title", "tags"]]
df_test["tags"] = df_test["tags"].apply(lambda x: ast.literal_eval(x))
df_test["doc_id"] = df_test.index
df_test = df_test.explode("tags")

top_tags = list(df_test["tags"].value_counts().head(10)[1:].index)
df_test = df_test[df_test["tags"].isin(top_tags)]
df_test = df_test.drop_duplicates("doc_id", keep="first")
df_test = df_test[~df_test["tags"].isna()]
df_test = df_test.sample(1000, random_state=42)

docs = df_test["title"].tolist()
ids = df_test["doc_id"].tolist()
tags = df_test["tags"].tolist()
metadata = {"tags": tags}


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
        embedding_model = SentenceTransformer(
            model_name_or_path="all-MiniLM-L6-v2", device=device
        )
        cls.bunka = Bunka(
            projection_model=projection_model, embedding_model=embedding_model
        )
        # metadata = None
        cls.bunka.fit(
            ids=ids,
            docs=docs,
            metadata=metadata,
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

        embedding_model = SentenceTransformer(
            model_name_or_path="all-MiniLM-L6-v2", device=device
        )

        bunka = Bunka(
            projection_model=projection_model, embedding_model=embedding_model
        )

        print("Fitting Bunka with SentenceTransformer")

        bunka.fit(
            ids=ids,
            docs=docs,
            metadata=metadata,
            pre_computed_embeddings=None,
            sampling_size_for_terms=1000,
        )

    def test_embed_flag_embeddings(self):

        projection_model = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=3,
            random_state=42,
        )

        embedding_model = FlagModel("BAAI/bge-small-en")

        bunka = Bunka(
            projection_model=projection_model, embedding_model=embedding_model
        )
        print("Fitting Bunka with FlagModel")
        bunka.fit(
            ids=ids,
            docs=docs,
            metadata=metadata,
            pre_computed_embeddings=None,
            sampling_size_for_terms=1000,
        )

        self.assertIsInstance(bunka, Bunka)

    def test_embed_hf_embed(self):

        projection_model = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=3,
            random_state=42,
        )

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        bunka = Bunka(
            projection_model=projection_model, embedding_model=embedding_model
        )

        print("Fitting Bunka with HuggingFaceEmbeddings")
        bunka.fit(
            ids=ids,
            docs=docs,
            metadata=metadata,
            pre_computed_embeddings=None,
            sampling_size_for_terms=1000,
        )

    # def test_topic_modeling_dbscan(self):
    #     custom_clustering_model = DBSCAN(eps=0.5, min_samples=5)

    #     df_topics = self.bunka.get_topics(
    #         custom_clustering_model=custom_clustering_model,
    #         n_clusters=10,
    #         min_count_terms=2,
    #         min_docs_per_cluster=30,
    #     )

    #     # df_topics_clean = self.bunka.get_clean_topic_name(llm=llm)
    #     self.assertIsInstance(df_topics, pd.DataFrame)
    #     self.assertIsInstance(self.bunka.df_top_docs_per_topic_, pd.DataFrame)

    def test_topic_modeling_kmeans(self):

        custom_clustering_model = KMeans(n_clusters=15)

        df_topics = self.bunka.get_topics(
            custom_clustering_model=custom_clustering_model,
            n_clusters=10,
            min_count_terms=2,
            min_docs_per_cluster=30,
        )

        # df_topics_clean = self.bunka.get_clean_topic_name(llm=llm)
        self.assertIsInstance(df_topics, pd.DataFrame)
        self.assertIsInstance(self.bunka.df_top_docs_per_topic_, pd.DataFrame)

    def test_visualize_topics(self):

        # Visualize Topics
        topic_fig = self.bunka.visualize_topics(
            width=800,
            height=800,
            show_text=False,
            density=True,
            colorscale="Portland",
            convex_hull=True,
            color=None,
        )
        if figure:
            topic_fig.show()

        self.assertIsInstance(topic_fig, go.Figure)

    def test_visualize_topics_colors(self):

        # Visualize Topics
        topic_fig = self.bunka.visualize_topics(
            width=800,
            height=800,
            show_text=True,
            density=True,
            colorscale="Portland",
            convex_hull=True,
            color="tags",
        )
        if figure:
            topic_fig.show()

        self.assertIsInstance(topic_fig, go.Figure)

    # def test_generative_names(self):
    #     n_clusters = 3
    #     self.bunka.get_topics(n_clusters=n_clusters, min_count_terms=1)
    #     df_topics_clean = self.bunka.get_clean_topic_name(llm=llm)
    #     print(df_topics_clean["topic_name"])
    #     self.assertIsInstance(df_topics_clean, pd.DataFrame)
    #     self.assertEqual(len(df_topics_clean), n_clusters)

    def test_bourdieu_modeling(self):
        bourdieu_fig = self.bunka.visualize_bourdieu(
            llm=None,
            x_left_words=["past"],
            x_right_words=["future"],
            y_top_words=["men"],
            y_bottom_words=["women"],
            height=800,
            width=800,
            clustering=True,
            topic_n_clusters=30,
            min_docs_per_cluster=50,
            density=False,
            colorscale="Portland",
        )
        if figure:
            bourdieu_fig.show()
        self.assertIsInstance(bourdieu_fig, go.Figure)

    # def test_save_bunka(self):
    #     self.bunka.save_bunka("bunka_dump")

    # def test_load_bunka(self):
    #     self.bunka.load_bunka("bunka_dump")


if __name__ == "__main__":
    unittest.main()
