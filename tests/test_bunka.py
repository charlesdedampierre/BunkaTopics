import os
import random
import sys
import unittest

import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub

from bunkatopics import Bunka

sys.path.append("../")

load_dotenv()

random.seed(42)


class TestBunka(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a sample dataset
        dataset = load_dataset("rguo123/trump_tweets")
        docs = dataset["train"]["content"]
        docs = random.sample(docs, 200)
        cls.bunka = Bunka()
        cls.bunka.fit(docs)

    def test_topic_modeling(self):
        # Test Topic Modeling
        n_clusters = 3
        df_topics = self.bunka.get_topics(n_clusters=n_clusters, min_count_terms=1)
        print(df_topics.name)
        self.assertIsInstance(df_topics, pd.DataFrame)
        self.assertEqual(len(df_topics), n_clusters)

        # Visualize Topics
        topic_fig = self.bunka.visualize_topics(width=800, height=800, show_text=True)
        self.assertIsInstance(topic_fig, go.Figure)

    def test_bourdieu_modeling(self):
        bourdieu_fig = self.bunka.visualize_bourdieu(
            x_left_words=["past"],
            x_right_words=["future"],
            y_top_words=["men"],
            y_bottom_words=["women"],
            height=800,
            width=800,
            clustering=False,
            topic_gen_name=False,
            topic_n_clusters=3,
        )
        # bourdieu_fig.show()
        self.assertIsInstance(bourdieu_fig, go.Figure)

    def test_generative_names(self):
        n_clusters = 3
        repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
        )

        self.bunka.get_topics(n_clusters=n_clusters, min_count_terms=1)
        df_topics_clean = self.bunka.get_clean_topic_name(llm=llm)
        print(df_topics_clean.name)
        self.assertIsInstance(df_topics_clean, pd.DataFrame)
        self.assertEqual(len(df_topics_clean), n_clusters)

        """
        # test Undimentional Map
        fig_solo = self.bunka.visualize_bourdieu_one_dimension(
            left=["negative", "bad"],
            right=["positive"],
            width=600,
            height=600,
            explainer=False,
        )
        self.assertIsInstance(fig_solo, go.Figure)

        # test RAG
        top_doc_len = 3
        res = self.bunka.rag_query(
            query="What are the main fight of Donald Trump ?",
            generative_model=open_ai_generative_model,
            top_doc=top_doc_len,
        )

        result = res["result"]
        self.assertIsInstance(result, str)

        document_sources = res["source_documents"]
        self.assertEqual(len(document_sources), top_doc_len)"""


if __name__ == "__main__":
    unittest.main()
