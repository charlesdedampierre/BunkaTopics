import plotly.graph_objs as go


def get_density_plot(
    x, y, x_centroids, y_centroids, label_centroids, width, height, marker_size=5
):

    fig_density = go.Figure(go.Histogram2dContour(x=x, y=y, colorscale="delta"))

    fig_density.update_traces(contours_coloring="fill", contours_showlabels=False)

    fig_density.update_layout(
        font_size=25,
        width=width,
        height=height,
        margin=dict(t=200),
        title=dict(font=dict(size=width / 40)),
    )

    # Add points with information
    fig_density.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="rgba(0,0,0,0.3)", size=marker_size),
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
            font=dict(family="Courier New, monospace", size=20, color="#ffffff"),
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=1,
            arrowcolor="#ff7f0e",
        )

    fig_density.update_layout(showlegend=False)

    return fig_density
