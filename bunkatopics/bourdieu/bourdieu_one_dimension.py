import random
import typing as t

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.embeddings import Embeddings

from bunkatopics.bourdieu.bourdieu_api import _get_continuum
from bunkatopics.datamodel import Document
from bunkatopics.visualization.visualization_utils import wrap_by_word

pd.options.mode.chained_assignment = None


class BourdieuOneDimensionVisualizer:
    """
    A class to visualize data distribution along a unique continuum inspired by Bourdieu's theory using an embedding model.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        left: str = ["aggressivity"],
        right: str = ["peacefulness"],
        height=700,
        width=600,
        explainer: bool = False,
        explainer_ngrams: list = [1, 2],
    ) -> None:
        """
        Constructs all the necessary attributes for the BourdieuOneDimensionVisualizer object.

        Args:
            embedding_model: The embedding model used for encoding text.
            left (List[str]): Keywords indicating one end of the continuum. Defaults to ["aggressivity"].
            right (List[str]): Keywords indicating the other end of the continuum. Defaults to ["peacefulness"].
            height (int): Height of the visualization plot. Default is 700.
            width (int): Width of the visualization plot. Default is 600.
            explainer (bool): If True, includes an explanation component in the visualization. Default is False.
            explainer_ngrams (List[int]): N-grams to use for generating explanations. Default is [1, 2].
        """
        pass
        self.embedding_model = embedding_model
        self.left = left
        self.right = right
        self.height = height
        self.width = width
        self.explainer = explainer
        self.explainer_ngrams = explainer_ngrams

    def fit_transform(self, docs: t.List[Document]) -> go.Figure:
        """
        Analyzes a list of Document objects and visualizes their distribution along the unique continuum.

        Args:
            docs (List[Document]): A list of Document objects to be analyzed.

        Returns:
            Tuple[go.Figure, plt]: A tuple containing a Plotly figures and a matplolib Figure. The first figure represents the
                                         distribution of data along the continuum, and the second figure
                                         (if explainer is True) represents specific terms that characterize
                                         the distribution.
        """
        self.id = str(random.randint(0, 10000))
        self.docs = docs

        self.new_docs = _get_continuum(
            embedding_model=self.embedding_model,
            docs=self.docs,
            cont_name=self.id,
            left_words=self.left,
            right_words=self.right,
            scale=False,
        )

        fig = self.plot_unique_dimension()
        return fig

    def plot_unique_dimension(self) -> go.Figure:
        """
        Generates a Plotly figure representing the unique dimension continuum.

        This method is used internally by fit_transform to create the visualization.

        Returns:
            go.Figure: A Plotly figure visualizing the distribution of documents along the unique continuum.
        """
        left = " ".join(self.left)
        right = " ".join(self.right)

        distances = [
            x.distance
            for doc in self.new_docs
            for x in doc.bourdieu_dimensions
            if x.continuum.id == self.id
        ]
        doc_id = [x.doc_id for x in self.new_docs]
        content = [x.content for x in self.new_docs]

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
            height=self.height,
            width=self.width,
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

        """fig_specific_terms = plot_specific_terms(
            docs=self.new_docs,
            left_words=left,
            right_words=right,
            id=self.id,
            ngrams=self.explainer_ngrams,
            quantile=0.80,
            top_n=20,
        )"""

        return fig
