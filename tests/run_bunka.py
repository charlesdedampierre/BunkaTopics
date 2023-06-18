from bunkatopics import Bunka
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.datasets import fetch_20newsgroups
import random

if __name__ == "__main__":
    full_docs = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )["data"]
    full_docs = random.sample(full_docs, 500)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    bunka = Bunka(model_hf=embedding_model)

    bunka.fit(full_docs)
    df_topics = bunka.get_topics(n_clusters=8)

    topic_fig = bunka.visualize_topics(width=800, height=800)
    topic_fig.show()
