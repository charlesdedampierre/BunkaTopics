import typing as t

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bunkatopics.datamodel import Document, Topic
from bunkatopics.visualization.visualization_utils import wrap_by_word

pd.options.mode.chained_assignment = None


class BourdieuVisualizer:
    """
    A class for visualizing Bourdieu's field analysis through a scatter plot.

    This visualizer plots documents on a 2D space based on their coordinates derived from
    Bourdieu's field analysis. It offers features like displaying percentage areas,
    drawing convex hulls around clusters, and labeling axes with specified terms.
    """

    def __init__(
        self,
        display_percent: bool = True,
        convex_hull: bool = True,
        clustering: bool = True,
        width: int = 800,
        height: int = 800,
        label_size_ratio_clusters: int = 100,
        label_size_ratio_label: int = 50,
        label_size_ratio_percent: int = 10,
        manual_axis_name: t.Optional[dict] = None,
        density: bool = True,
        colorscale: str = "delta",
    ) -> None:
        """
        Constructs all the necessary attributes for the BourdieuVisualizer object.

        Args:
            display_percent (bool): If True, display the percentage of documents in each quadrant.
            convex_hull (bool): If True, draw convex hulls around topic clusters.
            clustering (bool): If True, enable clustering of topics.
            width (int): Width of the plot.
            height (int): Height of the plot.
            label_size_ratio_clusters (int): Size ratio for the labels of clusters.
            label_size_ratio_label (int): Size ratio for the labels on axes.
            label_size_ratio_percent (int): Size ratio for the percentage labels.
            manual_axis_name (Optional[dict]): Custom names for the axes, if provided.
            colorscale (str): The color scale for contour density representation. Defaults to "delta".
            density (bool): Whether to display a density map
        """
        self.display_percent = display_percent
        self.convex_hull = convex_hull
        self.clustering = clustering
        self.width = width
        self.height = height
        self.label_size_ratio_clusters = label_size_ratio_clusters
        self.label_size_ratio_label = label_size_ratio_label
        self.label_size_ratio_percent = label_size_ratio_percent
        self.manual_axis_name = manual_axis_name
        self.density = density
        self.colorscale = colorscale

    def fit_transform(
        self, bourdieu_docs: t.List[Document], bourdieu_topics: t.List[Topic]
    ) -> go.Figure:
        """
        Transforms the given documents and topics into a Plotly figure based on Bourdieu's analysis.

        This method takes a list of documents and topics, processes them, and returns a Plotly figure
        representing the documents in a 2D space as per Bourdieu's field analysis.

        Args:
            bourdieu_docs (List[Document]): A list of Document objects to be visualized.
            bourdieu_topics (List[Topic]): A list of Topic objects used for clustering.

        Returns:
            go.Figure: A Plotly figure object representing the visualized documents and topics.
        """
        df_fig = pd.DataFrame(
            {
                "doc_id": [x.doc_id for x in bourdieu_docs],
                "x": [x.x for x in bourdieu_docs],
                "y": [x.y for x in bourdieu_docs],
                "content": [x.content for x in bourdieu_docs],
            }
        )

        df_fig["Text"] = df_fig["content"].apply(lambda x: wrap_by_word(x, 10))

        if self.density:
            fig = go.Figure(
                go.Histogram2dContour(
                    x=df_fig["x"],
                    y=df_fig["y"],
                    colorscale=self.colorscale,
                    showscale=False,
                    hoverinfo="none",
                ),
            )
        else:
            fig = go.Figure()

        scatter_fig = px.scatter(
            df_fig,
            x="x",
            y="y",
            # color="outside",
            # color_discrete_map={"1": "white", "0": "grey"},
            # hover_data=["Text"],
            hover_data={"x": False, "y": False, "Text": True},
            template="simple_white",
            opacity=0.3,
        )

        for trace in scatter_fig.data:
            fig.add_trace(trace)

        if self.density:
            label_axe_color = "white"
        else:
            label_axe_color = "black"

        fig.update_xaxes(
            title_text="",
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            showticklabels=False,
            zeroline=True,
            zerolinecolor=label_axe_color,
            zerolinewidth=3,
        )
        fig.update_yaxes(
            title_text="",
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            showticklabels=False,
            zeroline=True,
            zerolinecolor=label_axe_color,
            zerolinewidth=3,
        )

        # Add axis lines for x=0 and y=0
        fig.add_shape(
            type="line",
            x0=0,
            x1=0,
            y0=min(df_fig["y"]),
            y1=max(df_fig["y"]),
            line=dict(color=label_axe_color, width=3),  # Customize line color and width
        )

        fig.add_shape(
            type="line",
            x0=min(df_fig["x"]),
            x1=max(df_fig["x"]),
            y0=0,
            y1=0,
            line=dict(color=label_axe_color, width=3),  # Customize line color and width
        )

        x_left_name = bourdieu_docs[0].bourdieu_dimensions[0].continuum.left_words
        x_left_name = " ".join(x_left_name)

        x_right_name = bourdieu_docs[0].bourdieu_dimensions[0].continuum.right_words
        x_right_name = " ".join(x_right_name)

        y_top_name = bourdieu_docs[0].bourdieu_dimensions[1].continuum.left_words
        y_top_name = " ".join(y_top_name)

        y_bottom_name = bourdieu_docs[0].bourdieu_dimensions[1].continuum.right_words
        y_bottom_name = " ".join(y_bottom_name)

        if self.manual_axis_name is not None:
            y_top_name = self.manual_axis_name["y_top_name"]
            y_bottom_name = self.manual_axis_name["y_bottom_name"]
            x_left_name = self.manual_axis_name["x_left_name"]
            x_right_name = self.manual_axis_name["x_right_name"]

        fig.update_layout(
            annotations=[
                dict(
                    x=0,
                    # y=max_val,
                    y=max(df_fig["y"]),
                    xref="x",
                    yref="y",
                    text=y_top_name,
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(
                        size=self.width / self.label_size_ratio_label,
                        color=label_axe_color,
                    ),
                ),
                dict(
                    x=0,
                    y=min(df_fig["y"]),
                    # y=-max_val,
                    xref="x",
                    yref="y",
                    text=y_bottom_name,
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(
                        size=self.width / self.label_size_ratio_label,
                        color=label_axe_color,
                    ),
                ),
                dict(
                    x=max(df_fig["x"]),
                    # x=max_val,
                    y=0,
                    xref="x",
                    yref="y",
                    text=x_left_name,
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    font=dict(
                        size=self.width / self.label_size_ratio_label,
                        color=label_axe_color,
                    ),
                ),
                dict(
                    x=min(df_fig["x"]),
                    # x=-max_val,
                    y=0,
                    xref="x",
                    yref="y",
                    text=x_right_name,
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(
                        size=self.width / self.label_size_ratio_label,
                        color=label_axe_color,
                    ),
                ),
            ]
        )
        if self.clustering:
            topics_x = [x.x_centroid for x in bourdieu_topics]
            topics_y = [x.y_centroid for x in bourdieu_topics]
            topic_names = [x.name for x in bourdieu_topics]
            topics_name_plotly = [wrap_by_word(x, 7) for x in topic_names]

            # Display Topics
            for x, y, label in zip(topics_x, topics_y, topics_name_plotly):
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=label,
                    font=dict(
                        family="Courier New, monospace",
                        size=self.width / self.label_size_ratio_clusters,
                        color="blue",
                    ),
                    bordercolor="#c7c7c7",
                    borderwidth=self.width / 1000,
                    borderpad=self.width / 500,
                    bgcolor="white",
                    opacity=1,
                )

            if self.convex_hull:
                try:
                    for topic in bourdieu_topics:
                        # Create a Scatter plot with the convex hull coordinates
                        trace = go.Scatter(
                            x=topic.convex_hull.x_coordinates,
                            y=topic.convex_hull.y_coordinates,  # Assuming y=0 for simplicity
                            mode="lines",
                            name="Convex Hull",
                            line=dict(color="grey", dash="dot"),
                            showlegend=False,
                            hoverinfo="none",
                        )

                        fig.add_trace(trace)
                except Exception as e:
                    print(e)

        if self.display_percent:
            # Calculate the percentage for every box
            opacity = 0.4
            case1_count = len(df_fig[(df_fig["x"] < 0) & (df_fig["y"] < 0)])
            total_count = len(df_fig)
            case1_percentage = str(round((case1_count / total_count) * 100, 1)) + "%"

            fig.add_annotation(
                x=min(df_fig["x"]),
                y=min(df_fig["y"]),
                text=case1_percentage,
                font=dict(
                    family="Courier New, monospace",
                    size=self.width / self.label_size_ratio_percent,
                    color="grey",
                ),
                opacity=opacity,
                xanchor="left",
            )

            case2_count = len(df_fig[(df_fig["x"] < 0) & (df_fig["y"] > 0)])
            case2_percentage = str(round((case2_count / total_count) * 100, 1)) + "%"

            fig.add_annotation(
                x=min(df_fig["x"]),
                y=max(df_fig["y"]),
                text=case2_percentage,
                font=dict(
                    family="Courier New, monospace",
                    size=self.width / self.label_size_ratio_percent,
                    color="grey",
                ),
                opacity=opacity,
                xanchor="left",
            )

            case3_count = len(df_fig[(df_fig["x"] > 0) & (df_fig["y"] < 0)])
            case3_percentage = str(round((case3_count / total_count) * 100, 1)) + "%"

            fig.add_annotation(
                x=max(df_fig["x"]),
                y=min(df_fig["y"]),
                text=case3_percentage,
                font=dict(
                    family="Courier New, monospace",
                    size=self.width / self.label_size_ratio_percent,
                    color="grey",
                ),
                opacity=opacity,
                xanchor="left",
            )

            case4_count = len(df_fig[(df_fig["x"] > 0) & (df_fig["y"] > 0)])
            case4_percentage = str(round((case4_count / total_count) * 100, 1)) + "%"

            fig.add_annotation(
                x=max(df_fig["x"]),
                y=max(df_fig["y"]),
                text=case4_percentage,
                font=dict(
                    family="Courier New, monospace",
                    size=self.width / self.label_size_ratio_percent,
                    color="grey",
                ),
                opacity=opacity,
                xanchor="left",
            )

        fig.update_layout(
            font_size=25,
            height=self.height,
            width=self.width,
            margin=dict(
                t=self.width / 50,
                b=self.width / 50,
                r=self.width / 50,
                l=self.width / 50,
            ),
        )

        fig.update_layout(showlegend=False)
        return fig
