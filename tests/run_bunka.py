import sys

sys.path.append("../")

from bunkatopics import Bunka
from langchain.embeddings import HuggingFaceEmbeddings
import random
from datasets import load_dataset

random.seed(42)

if __name__ == "__main__":
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]["title"]
    full_docs = random.sample(dataset, 500)
    bunka = Bunka(model_hf=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    bunka.fit(full_docs)
    df_topics = bunka.get_topics(n_clusters=8)

    topic_fig = bunka.visualize_topics(width=800, height=800)
    topic_fig.show()

    bourdieu_fig = bunka.visualize_bourdieu(
        x_left_words=["joy"],
        x_right_words=["fear"],
        y_top_words=["politics"],
        y_bottom_words=["business"],
        height=1500,
        width=1500,
        label_size_ratio_label=50,
        clustering=False,
        display_percent=True,
    )

    bourdieu_fig.show()

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

    dimension_fig.show()
