import random
import typing as t

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bunkatopics.bourdieu.bourdieu_api import _get_continuum
from bunkatopics.datamodel import Document
from bunkatopics.visualization.topic_explainer import plot_specific_terms
from bunkatopics.visualization.visualization_utils import wrap_by_word

pd.options.mode.chained_assignment = None


def visualize_bourdieu_one_dimension(
    docs: t.List[Document],
    embedding_model,
    left: str = ["aggressivity"],
    right: str = ["peacefulness"],
    height=700,
    width=600,
    explainer: bool = True,
    explainer_ngrams: list = [1, 2],
) -> go.Figure:
    """
    Visualize the distribution of data along a unique continuum inspired by Bourdieu's concept.

    Args:
        docs (List[Document]): A list of Document objects to analyze.
        embedding_model: The embedding model used for encoding text.
        left (str): Keywords indicating one end of the continuum (default is ["aggressivity"]).
        right (str): Keywords indicating the other end of the continuum (default is ["peacefulness"]).
        height (int): Height of the visualization plot (default is 700).
        width (int): Width of the visualization plot (default is 600).
        explainer (bool): Whether to include an explanation (default is True).
        explainer_ngrams (list): N-grams to use for generating explanations (default is [1, 2]).

    Returns:
        go.Figure: A Plotly figure displaying the distribution of data along the unique continuum.
    """
    id = str(random.randint(0, 10000))

    new_docs = _get_continuum(
        embedding_model=embedding_model,
        docs=docs,
        cont_name=id,
        left_words=left,
        right_words=right,
        scale=False,
    )

    fig, fig_specific_terms = plot_unique_dimension(
        new_docs,
        id=id,
        left=left,
        right=right,
        height=height,
        width=width,
        explainer=explainer,
        explainer_ngrams=explainer_ngrams,
    )
    return fig, fig_specific_terms


def plot_unique_dimension(
    docs: t.List[Document],
    id: str = id,
    left: list = ["aggressivity"],
    right: list = ["peacefullness"],
    height=700,
    width=600,
    explainer: bool = True,
    explainer_ngrams: list = [1, 2],
) -> go.Figure:
    left = " ".join(left)
    right = " ".join(right)

    distances = [
        x.distance
        for doc in docs
        for x in doc.bourdieu_dimensions
        if x.continuum.id == id
    ]
    doc_id = [x.doc_id for x in docs]
    content = [x.content for x in docs]

    df_distances = pd.DataFrame(
        {"doc_id": doc_id, "distances": distances, "content": content}
    )

    name = "<" + right + "-" + left + ">"

    df_fig = df_distances.rename(columns={"distances": name})
    df_fig["content"] = df_fig["content"].apply(lambda x: wrap_by_word(x, 10))

    fig = px.box(
        df_fig,
        y=name,
        points="all",
        hover_data=["content"],
        height=height,
        width=width,
        template="plotly_white",
    )

    fig.add_shape(
        dict(
            type="line",
            x0=df_fig[name].min(),  # Set the minimum x-coordinate of the line
            x1=df_fig[name].max(),  # Set the maximum x-coordinate of the line
            y0=0,
            y1=0,
            line=dict(color="red", width=4),
        )
    )

    fig_specific_terms = plot_specific_terms(
        docs=docs,
        left_words=left,
        right_words=right,
        id=id,
        ngrams=explainer_ngrams,
        quantile=0.80,
        top_n=20,
    )

    return (fig, fig_specific_terms)
