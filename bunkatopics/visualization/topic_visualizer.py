import typing as t

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from bunkatopics.datamodel import Document, Topic
from bunkatopics.visualization.visualization_utils import (check_list_type,
                                                           wrap_by_word)


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
        point_size_ratio=150,
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
        self.point_size_ratio = point_size_ratio
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

    def fit_transform(
        self,
        docs: t.List[Document],
        topics: t.List[Topic],
        color: str = None,
    ) -> go.Figure:
        """
        Generates a Plotly figure visualizing the given documents and topics.

        This method processes the documents and topics to create a 2D scatter plot,
        showing the distribution and clustering of topics. It supports displaying text labels,
        contour density, and centroids for topics.

        Args:
            docs (List[Document]): A list of Document objects to be visualized.
            topics (List[Topic]): A list of Topic objects for clustering visualization.
            color (str): The metadata field to use for coloring the documents. Defaults to None.

        Returns:
            go.Figure: A Plotly figure object representing the visualized documents and topics.
        """

        docs_x = [doc.x for doc in docs]
        docs_y = [doc.y for doc in docs]
        docs_topic_id = [doc.topic_id for doc in docs]
        docs_content = [doc.content for doc in docs]
        docs_content_plotly = [wrap_by_word(x, 10) for x in docs_content]

        # remove the topic names with no topics
        topics = [x for x in topics if x.topic_id != "bt-no-topic"]
        topics_x = [topic.x_centroid for topic in topics]
        topics_y = [topic.y_centroid for topic in topics]
        topics_name = [topic.name for topic in topics]
        topics_name_plotly = [wrap_by_word(x, 6) for x in topics_name]

        if color is not None:
            self.density = None

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

        if color is not None:
            list_color = [x.metadata[color] for x in docs]
            nk[:, 2] = np.array(list_color).reshape(-1, 1)
            hovertemplate = f"<br>%{{customdata[1]}}<br>{color}: %{{customdata[2]}}"
        else:
            hovertemplate = "<br>%{customdata[1]}<br>"

        def extend_color_palette(number_of_categories):
            list_of_colors = px.colors.qualitative.Dark24
            extended_list_of_colors = (
                list_of_colors * (number_of_categories // len(list_of_colors))
                + list_of_colors[: number_of_categories % len(list_of_colors)]
            )
            return extended_list_of_colors

        if color is not None:
            if len(list_color) > 24:
                list_of_colors = extend_color_palette(len(list_color))
            else:
                list_of_colors = px.colors.qualitative.Dark24

        if color is not None:
            if check_list_type(list_color) == "string":
                unique_categories = list(set(list_color))
                colormap = {
                    category: list_of_colors[i]
                    for i, category in enumerate(unique_categories)
                }
                list_color_figure = [colormap[value] for value in list_color]
                colorscale = None
                colorbar = None

            else:
                list_color_figure = list_color
                colorscale = "RdBu"
                colorbar = dict(title=color, tickfont=dict(size=self.width / 50))

        # if search is not None:
        #     from .visualization_utils import normalize_list

        #     docs_search = self.vectorstore.similarity_search_with_score(
        #         search, k=len(self.vectorstore.get()["documents"])
        #     )
        #     similarity_score = [doc[1] for doc in docs_search]
        #     similarity_score_norm = normalize_list(similarity_score)
        #     similarity_score_norm = [1 - doc for doc in similarity_score_norm]

        #     docs_search = {
        #         "doc_id": [doc[0].metadata["doc_id"] for doc in docs_search],
        #         "score": [score for score in similarity_score_norm],
        #         "page_content": [doc[0].page_content for doc in docs_search],
        #     }

        #     list_color_figure = docs_search["score"]
        #     colorscale = "RdBu"
        #     colorbar = dict(title="Semantic Similarity")

        else:
            list_color_figure = None
            colorscale = None
            colorbar = None

        if self.show_text:
            # Add points with information
            fig_density.add_trace(
                go.Scatter(
                    x=docs_x,
                    y=docs_y,
                    mode="markers",
                    marker=dict(
                        color=list_color_figure,  # Assigning colors based on the list_color
                        size=self.width / self.point_size_ratio,
                        # size=10,  # Adjust the size of the markers as needed
                        opacity=0.5,  # Adjust the opacity of the markers as needed
                        colorscale=colorscale,  # You can specify a colorscale if needed
                        colorbar=colorbar,  # Optional colorbar title
                    ),
                    showlegend=False,
                    customdata=nk,
                    hovertemplate=hovertemplate,
                ),
            )

        if color is not None:
            if check_list_type(list_color) == "string":
                # Create legend based on categories
                legend_items = []
                for category, color_item in colormap.items():
                    legend_items.append(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            marker=dict(color=color_item),
                            name=category,
                        )
                    )

                # Add legend items to the figure
                for item in legend_items:
                    fig_density.add_trace(item)

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
                        showlegend=False,
                    )
                    fig_density.add_trace(trace)
            except Exception as e:
                print(e)

        if color is not None:
            fig_density.update_layout(
                legend_title_text=color,
                legend=dict(
                    font=dict(
                        family="Arial",
                        size=int(self.width / 60),  # Adjust font size of the legend
                        color="black",
                    ),
                ),
            )

            fig_density.update_layout(plot_bgcolor="white")

        # fig_density.update_layout(showlegend=True)
        fig_density.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig_density.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig_density.update_yaxes(showticklabels=False)

        return fig_density
