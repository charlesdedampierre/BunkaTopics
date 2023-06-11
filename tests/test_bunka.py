import sys

sys.path.append("../")

import unittest
from bunkatopics import Bunka
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import fetch_20newsgroups
import random


class BunkaTestCase(unittest.TestCase):
    def setUp(self):
        docs = fetch_20newsgroups(
            subset="all", remove=("headers", "footers", "quotes")
        )["data"]
        docs = random.sample(docs, 100)
        self.bunka = Bunka()
        self.bunka.fit(docs)

    def pipeline(self):
        # Run the full pipelune
        n_clusters = 2
        df_topics = self.bunka.get_topics(n_clusters=n_clusters)

        # Add assertions to verify the expected output
        self.assertEqual(len(df_topics), n_clusters)
        self.assertIsInstance(df_topics, pd.DataFrame)

        bourdieu_fig = self.bunka.visualize_bourdieu(
            x_left_words=["past"],
            x_right_words=["future", "futuristis"],
            y_top_words=["politics", "Government"],
            y_bottom_words=["cultural phenomenons"],
            height=2000,
            width=2000,
            clustering=False,
        )

        self.assertIsInstance(bourdieu_fig, go.Figure)

        user_input = "this is a computer"
        results = self.bunka.search(user_input)

        # Add assertions to verify the expected output
        self.assertIsNotNone(results)
        self.assertIsInstance(results, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
