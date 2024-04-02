import os
import random
import unittest

import nbformat

# from nbconvert.preprocessors import ExecutePreprocessor
import traceback

from unittest.mock import patch
import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import HDBSCAN
import ast
import umap

load_dotenv()
from langchain_community.llms import HuggingFaceHub

from bunkatopics import Bunka

random.seed(42)

"""repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
)"""
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

docs = list(df_test["title"])
ids = list(df_test["doc_id"])
metadata = {"tags": list(df_test["tags"])}


class TestBunka(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a sample dataset

        # dataset = pd.read_csv(
        #     "/Users/charlesdedampierre/Desktop/Personal Data Science/meilli/imdb.csv"
        # )

        # dataset["iw"] = dataset["iw"].astype(str)

        # dataset = dataset[["imdb", "iw", "description", "avg_vote"]]

        # metadata = {'genre':list(data['genre']), 'iw':list(data['iw'])}
        # metadata = {"iw": list(dataset["iw"]), "avg_vote": list(dataset["avg_vote"])}

        dataset = load_dataset("rguo123/trump_tweets")
        docs = dataset["train"]["content"]
        docs = random.sample(docs, 5000)
        ids = None

        # from detoxify import Detoxify

        # print("Predicting toxicity...")
        # results = Detoxify("original").predict(docs)
        # metadata = {"toxicity": results["toxicity"]}

        from transformers import pipeline

        print("Sentiment Analysis...")

        sentiment_pipeline = pipeline("sentiment-analysis")
        # results = sentiment_pipeline(docs)
        # metadata = {"sentiment": [x["label"] for x in results]}

        from tqdm import tqdm

        # Create an empty list to store sentiment labels
        metadata = {"sentiment": []}

        # Iterate over the documents with tqdm to show the progress bar
        for doc in tqdm(docs, desc="Processing documents", unit="documents"):
            # Perform sentiment analysis on each document
            result = sentiment_pipeline(doc)
            # Append the sentiment label to the metadata
            metadata["sentiment"].append(result[0]["label"])

        projection_model = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=3,
            random_state=42,
        )

        # projection_model = umap.UMAP(
        #     n_components=2, n_neighbors=5, min_dist=0.3, random_state=42
        # )

        cls.bunka = Bunka(projection_model=projection_model)
        # metadata = None
        cls.bunka.fit(ids=ids, docs=docs, metadata=metadata)

    def test_topic_modeling(self):
        # Test Topic Modeling

        custom_clustering_model = KMeans(n_clusters=15)
        # custom_clustering_model = HDBSCAN()
        df_topics = self.bunka.get_topics(
            custom_clustering_model=custom_clustering_model,
            n_clusters=10,
            min_count_terms=2,
            min_docs_per_cluster=10,
        )
        self.assertIsInstance(df_topics, pd.DataFrame)
        # self.assertEqual(len(df_topics), n_clusters)

        # Visualize Topics
        topic_fig = self.bunka.visualize_topics(
            width=800,
            height=800,
            show_text=True,
            density=True,
            colorscale="Portland",
            convex_hull=True,
            # color="tags",
            color="sentiment",
        )
        if figure:
            topic_fig.show()

        self.assertIsInstance(topic_fig, go.Figure)
        self.assertIsInstance(self.bunka.df_top_docs_per_topic_, pd.DataFrame)
        print(self.bunka.df_top_docs_per_topic_)

    """def test_generative_names(self):
        n_clusters = 3
        self.bunka.get_topics(n_clusters=n_clusters, min_count_terms=1)
        df_topics_clean = self.bunka.get_clean_topic_name(llm=llm)
        print(df_topics_clean["topic_name"])
        self.assertIsInstance(df_topics_clean, pd.DataFrame)
        self.assertEqual(len(df_topics_clean), n_clusters)"""

    # def test_bourdieu_modeling(self):
    #     bourdieu_fig = self.bunka.visualize_bourdieu(
    #         llm=None,
    #         x_left_words=["past"],
    #         x_right_words=["future"],
    #         y_top_words=["men"],
    #         y_bottom_words=["women"],
    #         height=800,
    #         width=800,
    #         clustering=True,
    #         topic_n_clusters=30,
    #         min_docs_per_cluster=50,
    #         density=False,
    #         colorscale="Portland",
    #     )
    #     if figure:
    #         bourdieu_fig.show()
    #     self.assertIsInstance(bourdieu_fig, go.Figure)

    """def test_rag(self):
        top_doc_len = 3
        res = self.bunka.rag_query(
            query="What is great?",
            llm=None,
            top_doc=top_doc_len,
        )

        result = res["result"]
        print(result)
        self.assertIsInstance(result, str)
        document_sources = res["source_documents"]
        self.assertEqual(len(document_sources), top_doc_len)"""

    """def test_plot_query(self):
        query = "What is great?"
        fig_query, percent = self.bunka.visualize_query(
            query=query, width=800, height=800
        )
        self.assertIsInstance(fig_query, go.Figure)

    def test_boudieu_unique_dimension(self):
        fig_one_dimension = self.bunka.visualize_bourdieu_one_dimension(
            left=["negative"], right=["positive"], explainer=False
        )
        # fig_one_dimension.show()
        self.assertIsInstance(fig_one_dimension, go.Figure)"""

    """def test_topic_distribution(self):
        self.bunka.get_topics(n_clusters=3, min_count_terms=1)
        fig_distribution = self.bunka.get_topic_repartition()
        self.assertIsInstance(fig_distribution, go.Figure)"""

    """@patch("subprocess.run")
    @patch("bunkatopics.serveur.is_server_running", return_value=False)
    @patch("bunkatopics.serveur.kill_server")
    def test_bunka_server(
        self, mock_kill_server, mock_is_server_running, mock_subprocess_run
    ):
        # Ensure that start_server can be called without raising an exception
        self.bunka.get_topics(n_clusters=3, min_count_terms=1)
        try:
            self.bunka.start_server()
        except Exception as e:
            self.fail(f"start_server raised an exception: {e}")

        # Check if subprocess.run was called correctly
        mock_subprocess_run.assert_called_with(
            ["cp", "web/env.model", "web/.env"], check=True
        )
    """
    """def test_notebook(self):
        notebook_filename = (
            "notebooks/cleaning.ipynb"  # Replace with your notebook file
        )
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="bunka_kernel")

        try:
            ep.preprocess(nb)
        except Exception as e:
            print(f"Error executing the notebook {notebook_filename}.")
            print(e)
            traceback.print_exc()
            self.fail(f"Notebook {notebook_filename} failed to execute.")

    """


if __name__ == "__main__":
    unittest.main()
