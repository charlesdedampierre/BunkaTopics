import pandas as pd
from .visu_utils import wrap_by_word
from ..datamodel import Topic, ConvexHullModel, Document
from .convex_hull import get_convex_hull_coord
import numpy as np
import plotly.graph_objects as go
import typing as t


def visualize_topics(
    docs: t.List[Document], topics: t.List[Topic], width=1000, height=1000
):
    df_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
    try:
        convex_hull_list = []
        for i in set(df_docs["topic_id"]):
            points = df_docs[df_docs["topic_id"] == i][["x", "y"]].values

            # points =  points.reshape(-1, 1)

            x_ch, y_ch = get_convex_hull_coord(points, interpolate_curve=True)
            x_ch = list(x_ch)
            y_ch = list(y_ch)
            convex_hull_list.append(
                ConvexHullModel(topic_id=i, x_coordinates=x_ch, y_coordinates=y_ch)
            )
    except:
        pass

    docs_x = [doc.x for doc in docs]
    docs_y = [doc.y for doc in docs]
    docs_topic_id = [doc.topic_id for doc in docs]
    docs_content = [doc.content for doc in docs]
    docs_content_plotly = [wrap_by_word(x, 10) for x in docs_content]

    topics_x = [topic.x_centroid for topic in topics]
    topics_y = [topic.y_centroid for topic in topics]
    topics_name = [topic.name for topic in topics]
    topics_name_plotly = [wrap_by_word(x, 6) for x in topics_name]
    label_size_ratio = 100

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

    # Add points with information
    fig_density.add_trace(
        go.Scatter(
            x=docs_x,
            y=docs_y,
            mode="markers",
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
        for convex_hull in convex_hull_list:
            # Create a Scatter plot with the convex hull coordinates
            trace = go.Scatter(
                x=convex_hull.x_coordinates,
                y=convex_hull.y_coordinates,  # Assuming y=0 for simplicity
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
