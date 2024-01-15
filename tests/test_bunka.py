import sys

sys.path.append("../")
import random
import unittest

import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from dotenv import load_dotenv

from bunkatopics import Bunka

load_dotenv()

random.seed(42)


class TestBunka(unittest.TestCase):
    def setUp(self):
        # Load a sample dataset
        dataset = load_dataset("rguo123/trump_tweets")
        docs = dataset["train"]["content"]
        docs = random.sample(docs, 100)
        self.bunka = Bunka()
        self.bunka.fit(docs)

    def test_pipeline(self):
        # Test Topic Modeling
        n_clusters = 3
        df_topics = self.bunka.get_topics(n_clusters=n_clusters, min_count_terms=1)
        print(df_topics.name)
        self.assertIsInstance(df_topics, pd.DataFrame)
        self.assertEqual(len(df_topics), n_clusters)

        # Visualize Topics
        topic_fig = self.bunka.visualize_topics(width=800, height=800)
        self.assertIsInstance(topic_fig, go.Figure)

    """ # test Bourdieu Map
        bourdieu_fig = self.bunka.visualize_bourdieu(
            generative_model=open_ai_generative_model,
            x_left_words=["past"],
            x_right_words=["future", "futuristic"],
            y_top_words=["politics", "Government"],
            y_bottom_words=["cultural phenomenons"],
            height=2000,
            width=2000,
            clustering=True,
            topic_gen_name=True,
            topic_n_clusters=2,
        )

        self.assertIsInstance(bourdieu_fig, go.Figure)

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
