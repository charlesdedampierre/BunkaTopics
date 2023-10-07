import sys

sys.path.append("../")

from bunkatopics import Bunka
from bunkatopics.functions.clean_text import clean_tweet
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
import random
from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

random.seed(42)

if __name__ == "__main__":
    # Social Data
    dataset = load_dataset("rguo123/trump_tweets")["train"]["content"]
    full_docs = random.sample(dataset, 5000)
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

    bourdieu_fig = bunka.visualize_bourdieu(
        generative_model=generative_model,
        x_left_words=["war"],
        x_right_words=["peace"],
        y_top_words=["men"],
        y_bottom_words=["women"],
        height=1500,
        width=1500,
        label_size_ratio_label=50,
        display_percent=True,
        clustering=True,
        topic_n_clusters=10,
        topic_terms=5,
        topic_top_terms_overall=500,
        topic_gen_name=True,
        convex_hull=True,
        radius_size=0.5,
    )

    bourdieu_fig.show()
