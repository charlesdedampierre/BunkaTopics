import typing as t

import numpy as np
import plotly.graph_objects as go

from bunkatopics.datamodel import Document, Topic
from bunkatopics.visualization.visualization_utils import wrap_by_word


class TopicVisualizer:

    """
    A class for visualizing topics and their associated documents in a 2D density Map.

    This visualizer plots documents and topics on a 2D space with an option to show text labels,
    contour density representations, and topic centroids. The visualization is useful for
    understanding the distribution and clustering of topics in a document corpus.
    """

    def __init__(
        self,
        show_text=False,
        width=1000,
        height=1000,
        label_size_ratio=100,
        colorscale="delta",
        density: bool = False,
        convex_hull: bool = False,
    ) -> None:
        """
        Initializes the TopicVisualizer with specified parameters.

        Args:
            show_text (bool): If True, text labels are displayed on the plot. Defaults to False.
            width (int): The width of the plot in pixels. Defaults to 1000.
            height (int): The height of the plot in pixels. Defaults to 1000.
            label_size_ratio (int): The size ratio for label text. Defaults to 100.
            colorscale (str): The color scale for contour density representation. Defaults to "delta".
            density (bool): Whether to display a density map
            convex_hull (bool): Whether to display lines around the clusters
        """
        self.show_text = show_text
        self.width = width
        self.height = height
        self.label_size_ratio = label_size_ratio
        self.colorscale = colorscale
        self.density = density
        self.convex_hull = convex_hull

        self.colorscale_list = [
            "Greys",
            "YlGnBu",
            "Greens",
            "YlOrRd",
            "Bluered",
            "RdBu",
            "Reds",
            "Blues",
            "Picnic",
            "Rainbow",
            "Portland",
            "Jet",
            "Hot",
            "Blackbody",
            "Earth",
            "Electric",
            "Viridis",
            "Cividis",
            "Inferno",
            "Magma",
            "Plasma",
        ]

    def fit_transform(self, docs: t.List[Document], topics: t.List[Topic]) -> go.Figure:
        """
        Generates a Plotly figure visualizing the given documents and topics.

        This method processes the documents and topics to create a 2D scatter plot,
        showing the distribution and clustering of topics. It supports displaying text labels,
        contour density, and centroids for topics.

        Args:
            docs (List[Document]): A list of Document objects to be visualized.
            topics (List[Topic]): A list of Topic objects for clustering visualization.

        Returns:
            go.Figure: A Plotly figure object representing the visualized documents and topics.
        """
        # Extract data from documents and topics
        docs_x = [doc.x for doc in docs]
        docs_y = [doc.y for doc in docs]
        docs_topic_id = [doc.topic_id for doc in docs]
        docs_content = [doc.content for doc in docs]
        docs_content_plotly = [wrap_by_word(x, 10) for x in docs_content]

        topics_x = [topic.x_centroid for topic in topics]
        topics_y = [topic.y_centroid for topic in topics]
        topics_name = [topic.name for topic in topics]
        topics_name_plotly = [wrap_by_word(x, 6) for x in topics_name]

        if self.density:
            # Create a figure with Histogram2dContour
            fig_density = go.Figure(
                go.Histogram2dContour(
                    x=docs_x,
                    y=docs_y,
                    colorscale=self.colorscale,
                    showscale=False,
                    hoverinfo="none",
                )
            )

            fig_density.update_traces(
                contours_coloring="fill", contours_showlabels=False
            )

        else:
            fig_density = go.Figure()

        # Update layout settings
        fig_density.update_layout(
            font_size=25,
            width=self.width,
            height=self.height,
            margin=dict(
                t=self.width / 50,
                b=self.width / 50,
                r=self.width / 50,
                l=self.width / 50,
            ),
            title=dict(font=dict(size=self.width / 40)),
        )

        nk = np.empty(shape=(len(docs_content), 3, 1), dtype="object")
        nk[:, 0] = np.array(docs_topic_id).reshape(-1, 1)
        nk[:, 1] = np.array(docs_content_plotly).reshape(-1, 1)

        if self.show_text:
            # Add points with information
            fig_density.add_trace(
                go.Scatter(
                    x=docs_x,
                    y=docs_y,
                    mode="markers",
                    marker=dict(opacity=0.5),
                    customdata=nk,
                    hovertemplate=("<br>%{customdata[1]}<br>"),
                )
            )

        # Add centroids labels
        for x, y, label in zip(topics_x, topics_y, topics_name_plotly):
            fig_density.add_annotation(
                x=x,
                y=y,
                text=label,
                showarrow=True,
                arrowhead=1,
                font=dict(
                    family="Courier New, monospace",
                    size=self.width / self.label_size_ratio,
                    color="blue",
                ),
                bordercolor="#c7c7c7",
                borderwidth=self.width / 1000,
                borderpad=self.width / 500,
                bgcolor="white",
                opacity=1,
                arrowcolor="#ff7f0e",
            )

        if self.convex_hull:
            try:
                for topic in topics:
                    # Create a Scatter plot with the convex hull coordinates
                    trace = go.Scatter(
                        x=topic.convex_hull.x_coordinates,
                        y=topic.convex_hull.y_coordinates,
                        mode="lines",
                        name="Convex Hull",
                        line=dict(color="grey", dash="dot"),
                        hoverinfo="none",
                    )
                    fig_density.add_trace(trace)
            except Exception as e:
                print(e)

        fig_density.update_layout(showlegend=False)
        fig_density.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig_density.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig_density.update_yaxes(showticklabels=False)

        return fig_density
