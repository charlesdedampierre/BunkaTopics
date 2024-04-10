from bunkatopics import Bunka

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")

from sklearn.manifold import TSNE

projection_model = TSNE(
    n_components=2,
    learning_rate="auto",
    init="random",
    perplexity=3,
    random_state=42,
)

from datasets import load_dataset

docs = load_dataset("bunkalab/medium-sample-technology")["train"]["title"]


# Initialize Bunka with your chosen model and language preference
bunka = Bunka(
    embedding_model=embedding_model,
    projection_model=projection_model,
    language="english",
)  # You can choose any language you prefer

# Fit Bunka to your text data
bunka.fit(docs)


## Bourdieu Fig
bourdieu_fig = bunka.visualize_bourdieu(
    llm=None,
    x_left_words=["This is about business"],
    x_right_words=["This is about politics"],
    y_top_words=["this is about startups"],
    y_bottom_words=["This is about governments"],
    height=800,
    width=800,
    clustering=True,
    topic_n_clusters=2,
    density=False,
    convex_hull=True,
    radius_size=0.2,
    label_size_ratio_clusters=80,
)

bourdieu_fig.show()
