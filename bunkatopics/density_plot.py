import plotly.graph_objs as go
import numpy as np
from .utils import wrap_by_word


def get_density_plot(
    x,
    y,
    texts,
    clusters,
    x_centroids,
    y_centroids,
    label_centroids,
    width: int,
    height: int,
    sizes=None,
    colors=None,
):

    fig_density = go.Figure(
        go.Histogram2dContour(x=x, y=y, colorscale="delta", showscale=False)
    )

    # colorbar=None,

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

    texts = [wrap_by_word(x, 10) for x in texts]

    nk = np.empty(shape=(len(texts), 3, 1), dtype="object")
    nk[:, 0] = np.array(clusters).reshape(-1, 1)
    nk[:, 1] = np.array(texts).reshape(-1, 1)
    nk[:, 2] = np.array(sizes).reshape(-1, 1)

    # Add points with information
    fig_density.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=sizes, color=colors),
            # marker=dict(color="#000000"),
            customdata=nk,
            hovertemplate="<br><b>TOPIC</b>: %{customdata[0]}<br>"
            + "<br><b>TEXT</b>: %{customdata[1]}<br>"
            + "<br><b>SCORE</b>: %{customdata[2]}<br>",
        )
    )

    # Add centroids labels
    for x, y, label in zip(x_centroids, y_centroids, label_centroids):
        fig_density.add_annotation(
            x=x,
            y=y,
            text=label,
            showarrow=True,
            arrowhead=1,
            font=dict(
                family="Courier New, monospace", size=width / 100, color="#ffffff"
            ),
            bordercolor="#c7c7c7",
            borderwidth=width / 1000,
            borderpad=width / 500,
            bgcolor="#ff7f0e",
            opacity=1,
            arrowcolor="#ff7f0e",
        )

    fig_density.update_layout(showlegend=False)
    fig_density.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig_density.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig_density.update_yaxes(showticklabels=False)
    fig_density.update_layout(coloraxis_showscale=False)

    return fig_density
