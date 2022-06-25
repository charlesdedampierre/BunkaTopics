import plotly.graph_objs as go


def get_density_plot(x, y, width, height):

    fig_density = go.Figure(go.Histogram2dContour(x=x, y=y, colorscale="delta"))

    fig_density.update_traces(contours_coloring="fill", contours_showlabels=False)

    fig_density.update_layout(
        font_size=25,
        width=width,
        height=height,
        margin=dict(t=200),
        title=dict(font=dict(size=width / 40)),
    )

    fig_density.update_layout(showlegend=False)

    return fig_density
