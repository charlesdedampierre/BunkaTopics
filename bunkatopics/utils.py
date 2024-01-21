import typing as t

import pandas as pd

from bunkatopics.datamodel import Document, Topic


def _create_topic_dfs(topics: t.List[Topic], docs: t.List[Document]):
    df_topics = pd.DataFrame.from_records([topic.model_dump() for topic in topics])

    df_topics["percent"] = df_topics["size"] / df_topics["size"].sum()
    df_topics["percent"] = df_topics["percent"] * 100
    df_topics["percent"] = round(df_topics["percent"], 2)
    df_topics = df_topics.rename(columns={"name": "topic_name"})

    df_topics = df_topics[["topic_id", "topic_name", "size", "percent"].copy()]
    df_topics = df_topics.sort_values("size", ascending=False)
    df_topics = df_topics.reset_index(drop=True)

    # extract Dataframe for top documents per topic
    top_docs_topics = [x for x in docs if x.topic_ranking is not None]
    top_docs_topics = pd.DataFrame([x.model_dump() for x in top_docs_topics])
    top_docs_topics["ranking_per_topic"] = top_docs_topics["topic_ranking"].apply(
        lambda x: x.get("rank")
    )
    top_docs_topics = top_docs_topics[
        ["topic_id", "content", "ranking_per_topic", "doc_id"]
    ]
    top_docs_topics = top_docs_topics.sort_values(
        ["topic_id", "ranking_per_topic"], ascending=(True, True)
    )
    top_docs_topics = top_docs_topics.reset_index(drop=True)
    top_docs_topics = pd.merge(
        top_docs_topics, df_topics[["topic_id", "topic_name"]], on="topic_id"
    )

    return df_topics, top_docs_topics


class BunkaError(Exception):
    """Custom exception for Bunka-related errors."""

    pass
