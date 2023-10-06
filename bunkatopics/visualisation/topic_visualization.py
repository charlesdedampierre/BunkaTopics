import typing as t

import numpy as np
import plotly.graph_objects as go

from ..datamodel import Document, Topic
from .visu_utils import wrap_by_word


def visualize_topics(
    docs: t.List[Document],
    topics: t.List[Topic],
    add_scatter=False,
    width=1000,
    height=1000,
    label_size_ratio=100,
):
    docs_x = [doc.x for doc in docs]
    docs_y = [doc.y for doc in docs]
    docs_topic_id = [doc.topic_id for doc in docs]
    docs_content = [doc.content for doc in docs]
    docs_content_plotly = [wrap_by_word(x, 10) for x in docs_content]

    topics_x = [topic.x_centroid for topic in topics]
    topics_y = [topic.y_centroid for topic in topics]
    topics_name = [topic.name for topic in topics]
    topics_name_plotly = [wrap_by_word(x, 6) for x in topics_name]

    fig_density = go.Figure(
        go.Histogram2dContour(x=docs_x, y=docs_y, colorscale="delta", showscale=False)
    )

    fig_density.update_traces(contours_coloring="fill", contours_showlabels=False)

    fig_density.update_layout(
        font_size=25,
        width=width,
        height=height,
        margin=dict(
            t=width / 50,
            b=width / 50,
            r=width / 50,
            l=width / 50,
        ),
        title=dict(font=dict(size=width / 40)),
    )

    nk = np.empty(shape=(len(docs_content), 3, 1), dtype="object")
    nk[:, 0] = np.array(docs_topic_id).reshape(-1, 1)
    nk[:, 1] = np.array(docs_content_plotly).reshape(-1, 1)
    # nk[:, 2] = np.array(sizes).reshape(-1, 1)

    if add_scatter is True:
        # Add points with information
        fig_density.add_trace(
            go.Scatter(
                x=docs_x,
                y=docs_y,
                mode="markers",
                marker=dict(opacity=0.5),  # Set the alpha value here
                # marker=dict(size=sizes, color=colors),
                # marker=dict(color="#000000"),
                customdata=nk,
                hovertemplate="<br><b>TOPIC</b>: %{customdata[0]}<br>"
                + "<br><b>TEXT</b>: %{customdata[1]}<br>"
                + "<br><b>SCORE</b>: %{customdata[2]}<br>",
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
                size=width / label_size_ratio,
                # color="#ffffff",
                color="blue",
            ),
            bordercolor="#c7c7c7",
            borderwidth=width / 1000,
            borderpad=width / 500,
            # bgcolor="#ff7f0e",
            bgcolor="white",
            opacity=1,
            arrowcolor="#ff7f0e",
        )

    try:
        for topic in topics:
            # Create a Scatter plot with the convex hull coordinates
            trace = go.Scatter(
                x=topic.convex_hull.x_coordinates,
                y=topic.convex_hull.y_coordinates,  # Assuming y=0 for simplicity
                mode="lines",
                name="Convex Hull",
                line=dict(color="grey"),
            )

            fig_density.add_trace(trace)
    except:
        pass

    fig_density.update_layout(showlegend=False)
    fig_density.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig_density.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig_density.update_yaxes(showticklabels=False)

    return fig_density
