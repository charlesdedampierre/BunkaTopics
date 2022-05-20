import pandas as pd
from bunka_topics.basic_class import BasicSemantics


if __name__ == "__main__":

    df = pd.read_csv("data/imdb.csv", index_col=[0])
    df = df.sample(1000, random_state=42)
    df = df[["imdb", "description"]].dropna()

    # instantiate model
    model = BasicSemantics(df, index_var="imdb", text_var="description")

    model.fit(
        extract_terms=True,
        terms_embeddings=True,
        docs_embeddings=True,
        sample_size_terms=10000,
        terms_limit=3000,
        terms_ents=True,
        language="en",
    )

    terms = model.terms
