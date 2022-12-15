import pandas as pd
from bunkatopics import BunkaTopics
import plotly


if __name__ == "__main__":

    # data = pd.read_csv("dataset.csv")
    data = pd.read_csv(
        "/Users/charlesdedampierre/Desktop/bunka_superfunctions/tests/data/imdb.csv"
    )
    data = data.sample(200, random_state=42)

    # Instantiate the model, extract ther terms and Embed the documents

    model = BunkaTopics(
        data,  # dataFrame
        text_var="description",  # Text Columns
        index_var="imdb",  # Index Column (Mandatory)
        extract_terms=True,  # extract Terms ?
        terms_embeddings=False,  # extract terms Embeddings?
        docs_embeddings=True,  # extract Docs Embeddings?
        embeddings_model="distiluse-base-multilingual-cased-v1",  # Chose an embeddings Model
        multiprocessing=True,  # Multiprocessing of Embeddings
        language="en",  # Chose between English "en" and French "fr"
        sample_size_terms=len(data),
        terms_limit=10000,  # Top Terms to Output
        terms_ents=True,  # Extract entities
        terms_ngrams=(1, 2),  # Chose Ngrams to extract
        terms_ncs=True,  # Extract Noun Chunks
        terms_include_pos=["NOUN", "PROPN", "ADJ"],  # Include Part-of-Speech
        terms_include_types=["PERSON", "ORG"],
        reduction=2,
    )  # Include Entity Types

    topics = model.get_clusters(
        topic_number=20,
        top_terms=4,
        term_type="lemma",
        top_terms_included=1000,
        out_terms=None,
        ngrams=(1, 2),
    )

    fig = model.visualize_clusters(width=1000, height=1000)
    plotly.offline.plot(fig, filename="file.html")

    res = model.get_specific_documents_per_cluster(top_n=10)
    print(res)


    print(topics)

    # Create dictionaries
    dim1 = {"israel": ["love"]}
    dim2 = {"complot": ["bad"]}

    dictionnary = pd.DataFrame()

    for x in [dim1, dim2]:
        dictionnary = dictionnary.append(pd.DataFrame(x).unstack().reset_index())

    dictionnary = dictionnary.drop("level_1", 1).reset_index(drop=True)
    dictionnary.columns = ["category", "term"]

    df_folding, met = model.get_folding(dictionnary)

    print(met)
    print(df_folding)