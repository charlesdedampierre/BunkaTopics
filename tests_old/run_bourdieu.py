import sys

sys.path.append("../")

import os
import random

from datasets import load_dataset
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp

from bunkatopics import Bunka
from bunkatopics.functions.clean_text import clean_tweet

load_dotenv()

random.seed(42)

if __name__ == "__main__":
    # Social Data
    dataset = load_dataset("rguo123/trump_tweets")["train"]["content"]
    full_docs = random.sample(dataset, 3000)
    full_docs = [clean_tweet(x) for x in full_docs]

    # Scientific Litterature Data
    # dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]["title"]
    # full_docs = random.sample(dataset, 500)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bunka = Bunka(embedding_model=embedding_model)

    generative_model = LlamaCpp(
        model_path=os.getenv("MODEL_PATH"),
        n_ctx=2048,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        verbose=False,
    )
    generative_model.client.verbose = False

    bunka.fit(full_docs)

    manual_axis_name = {
        "x_left_name": "positive",
        "x_right_name": "negative",
        "y_top_name": "women",
        "y_bottom_name": "men",
    }

    bourdieu_fig = bunka.visualize_bourdieu(
        generative_model=generative_model,
        x_left_words=["this is a positive content"],
        x_right_words=["this is a negative content"],
        y_top_words=["this is about women"],
        y_bottom_words=["this is about men"],
        height=1000,
        width=1000,
        display_percent=True,
        use_doc_gen_topic=True,
        clustering=True,
        topic_n_clusters=10,
        topic_terms=5,
        topic_top_terms_overall=500,
        topic_gen_name=True,
        convex_hull=True,
        radius_size=0.5,
        manual_axis_name=manual_axis_name,
    )

    bourdieu_fig.show()
