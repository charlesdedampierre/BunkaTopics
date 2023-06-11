import sys

sys.path.append("../")

import unittest
from bunkatopics import Bunka
import pandas as pd


class BunkaTestCase(unittest.TestCase):
    def setUp(self):
        self.bunka = Bunka()
        docs = [
            "john is in the kitchen",
            "Therese is coding",
            "Charles is playing outside",
            "Someon is coming to the house",
        ]

        self.bunka.fit(docs)

    def test_fit_transform(self):
        n_clusters = 2
        df_topics = self.bunka.get_topics(n_clusters=n_clusters)

        # Add assertions to verify the expected output
        self.assertEqual(len(df_topics), n_clusters)
        self.assertIsInstance(df_topics, pd.DataFrame)

    def test_search(self):
        user_input = "this is a computer"
        results = self.bunka.search(user_input)

        # Add assertions to verify the expected output
        self.assertIsNotNone(results)
        self.assertIsInstance(results, pd.DataFrame)

    # Add more test methods for other functions in the Bunka class


if __name__ == "__main__":
    unittest.main()
