import sys

sys.path.append("../")

from bunkatopics import Bunka
from langchain.embeddings import HuggingFaceEmbeddings
import random
from datasets import load_dataset
from bunkatopics.functions.clean_text import clean_tweet
import openai
import os
from dotenv import load_dotenv

load_dotenv()

random.seed(42)

if __name__ == "__main__":
    dataset = load_dataset("rguo123/trump_tweets")["train"]["content"]
    full_docs = random.sample(dataset, 500)
    full_docs = [clean_tweet(x) for x in full_docs]
    # dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]["title"]
    # full_docs = random.sample(dataset, 500)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bunka = Bunka(model_hf=embedding_model)

    bunka.fit(full_docs)
    df_topics = bunka.get_topics(n_clusters=8)

    topic_fig = bunka.visualize_topics(width=800, height=800)
    topic_fig.show()

    bourdieu_fig = bunka.visualize_bourdieu(
        x_left_words=["war"],
        x_right_words=["peace"],
        y_top_words=["men"],
        y_bottom_words=["women"],
        openai_key=os.getenv("OPEN_AI_KEY"),
        height=1500,
        width=1500,
        label_size_ratio_label=50,
        display_percent=True,
        clustering=True,
        topic_n_clusters=10,
        topic_terms=5,
        topic_top_terms_overall=500,
        topic_gen_name=True,
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

    from bunkatopics.visualisation.bourdieu import visualize_bourdieu_one_dimension

    fig = visualize_bourdieu_one_dimension(
        docs=bunka.docs,
        embedding_model=embedding_model,
        left=["negative", "bad"],
        right=["positive"],
        width=1200,
        height=1200,
    )

    fig.show()
