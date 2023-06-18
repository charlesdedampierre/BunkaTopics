from bunkatopics import Bunka
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.datasets import fetch_20newsgroups
import random

random.seed(42)

if __name__ == "__main__":
    full_docs = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )["data"]
    full_docs = random.sample(full_docs, 100)
    full_docs = [x for x in full_docs if len(x) >= 50]  # Minimum lenght of texts

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    bunka = Bunka(model_hf=embedding_model)

    bunka.fit(full_docs)
    df_topics = bunka.get_topics(n_clusters=8)

    topic_fig = bunka.visualize_topics(width=800, height=800)
    topic_fig.show()

    bourdieu_fig = bunka.visualize_bourdieu(
        x_left_words=["past"],
        x_right_words=["future", "futuristic"],
        y_top_words=["politics", "Government"],
        y_bottom_words=["cultural phenomenons"],
        height=1500,
        width=1500,
        clustering=True,
    )

    bourdieu_fig.show()
