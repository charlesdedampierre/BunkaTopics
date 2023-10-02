import typing as t

import pandas as pd
import plotly.express as px

from bunkatopics.visualisation.visu_utils import wrap_by_word

from ..datamodel import Topic


def get_topic_repartition(topics: t.List[Topic], width=1200, height=800):
    df_topics = pd.DataFrame.from_records([topic.dict() for topic in topics])
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
