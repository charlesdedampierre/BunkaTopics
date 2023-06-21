import sys

sys.path.append("../")

from bunkatopics import Bunka
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.datasets import fetch_20newsgroups
import random

random.seed(42)

if __name__ == "__main__":
    full_docs = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )["data"]
    full_docs = random.sample(full_docs, 1000)
    full_docs = [x for x in full_docs if len(x) >= 50]  # Minimum lenght of texts

    # bunka = Bunka()

    bunka = Bunka(model_hf=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    bunka.fit(full_docs)
    df_topics = bunka.get_topics(n_clusters=8)

    topic_fig = bunka.visualize_topics(width=800, height=800)
    topic_fig.show()

    bourdieu_fig = bunka.visualize_bourdieu(
        x_left_words=["joy"],
        x_right_words=["fear"],
        y_top_words=["local politics"],
        y_bottom_words=["international politics"],
        height=1500,
        width=1500,
        clustering=False,
        display_percent=True,
    )

    dimensions = [
        "Happiness",
        "Sadness",
        "Anger",
        "Love",
        "Surprise",
        "Fear",
        "Excitement",
        "Disgust",
        "Confusion",
        "Gratitude",
    ]
    dimension_fig = bunka.get_dimensions(dimensions=dimensions, height=1500, width=1500)

    bourdieu_fig.show()
    dimension_fig.show()
