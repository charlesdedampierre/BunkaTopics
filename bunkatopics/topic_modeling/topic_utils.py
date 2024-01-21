import typing as t

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from bunkatopics.datamodel import Topic
from bunkatopics.visualization.visualization_utils import wrap_by_word


def get_topic_repartition(
    topics: t.List[Topic], width: int = 1200, height: int = 800
) -> Figure:
    """
    Create a bar plot to visualize the distribution of topics by size.

    Args:
        topics (List[Topic]): A list of Topic objects containing information about topics.
        width (int): Width of the visualization plot (default is 1200).
        height (int): Height of the visualization plot (default is 800).

    Returns:
        Figure: A Plotly figure displaying the distribution of topics by size in a bar plot.
    """
    df_topics = pd.DataFrame.from_records([topic.model_dump() for topic in topics])
    df_topics["name"] = df_topics["name"].apply(lambda x: wrap_by_word(x, 8))
    df_topics = df_topics.sort_values("size")

    fig = px.bar(
        df_topics,
        y="name",
        x="size",
        width=width,
        height=height,
        title="Topic Size",
        template="simple_white",
    )

    return fig
