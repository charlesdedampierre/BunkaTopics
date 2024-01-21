import typing as t

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

from bunkatopics.datamodel import Document
from bunkatopics.visualization.visualization_utils import wrap_by_word


def plot_query(
    embedding_model,
    docs: t.List[Document],
    query: str = "What is firearm?",
    min_score: float = 0.7,
    height: int = 600,
    width: int = 400,
) -> go.Figure:
    """
    Visualize the similarity scores between a query and a list of documents.

    Args:
        embedding_model: The embedding model used for encoding text.
        docs (List[Document]): A list of Document objects containing content and embeddings.
        query (str): The query text for which similarity scores are calculated (default is "What is firearm?").
        min_score (float): The minimum similarity score for including a document (default is 0.7).
        height (int): Height of the visualization plot (default is 600).
        width (int): Width of the visualization plot (default is 400).

    Returns:
        go.Figure: A Plotly figure displaying similarity scores and content for selected documents.
        float: The percentage of documents with similarity scores above the minimum score.
    """

    query_embedding = embedding_model.embed_query(query)

    ids = [x.doc_id for x in docs]
    contents = [x.content for x in docs]
    embeddings = [x.embedding for x in docs]
    embeddings_array = np.array(embeddings)
    similarities = cosine_similarity([query_embedding], embeddings_array)
    similarities = similarities.tolist()[0]

    df_unique = pd.DataFrame({"ids": ids, "score": similarities, "content": contents})
    df_unique = df_unique.sort_values("score", ascending=False).reset_index(drop=True)
    df_unique = df_unique[df_unique["score"] > min_score]
    df_unique["content"] = df_unique["content"].apply(lambda x: wrap_by_word(x, 10))

    percent = round(len(df_unique) / len(ids) * 100, 2)

    fig = px.box(
        df_unique,
        y="score",
        points="all",
        hover_data=["content"],
        height=height,
        width=width,
        template="plotly_white",
        orientation="v",
        title=str(percent) + "%",
        boxmode="overlay",
    )

    return fig, percent
