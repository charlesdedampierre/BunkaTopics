from bunkatopics import Bunka

import random
import sys
import unittest

import pandas as pd
import plotly.graph_objects as go
from datasets import load_dataset
from bunkatopics import Bunka

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
        topic_fig.show()
        self.assertIsInstance(topic_fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
