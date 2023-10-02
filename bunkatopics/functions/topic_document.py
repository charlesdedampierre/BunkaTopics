import typing as t

import pandas as pd

from ..datamodel import Document, Topic


def get_top_documents(
    docs: t.List[Document], topics: t.List[Topic], ranking_terms=20, top_docs=100
) -> t.List[Topic]:
    df_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_docs = df_docs[["doc_id", "topic_id", "term_id"]]
    df_docs = df_docs.explode("term_id").reset_index(drop=True)

    df_topics = pd.DataFrame.from_records([topic.dict() for topic in topics])
    df_topics["term_id"] = df_topics["term_id"].apply(lambda x: x[:ranking_terms])
    df_topics = df_topics[["topic_id", "term_id"]]
    df_topics = df_topics.explode("term_id").reset_index(drop=True)

    df_rank = pd.merge(df_docs, df_topics, on=["topic_id", "term_id"])
    df_rank = (
        df_rank.groupby(["topic_id", "doc_id"])["term_id"]
        .count()
        .rename("count_topic_terms")
        .reset_index()
    )
    df_rank = df_rank.sort_values(
        ["topic_id", "count_topic_terms"], ascending=(True, False)
    ).reset_index(drop=True)
    df_rank = df_rank.groupby("topic_id").head(5).reset_index(drop=True)

    df_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_docs = df_docs[["doc_id", "content"]]

    df_rank = pd.merge(df_rank, df_docs, on="doc_id")
    df_topics = pd.DataFrame.from_records([topic.dict() for topic in topics])
    df_topics = df_topics[["topic_id", "name"]]

    df_rank = pd.merge(df_rank, df_topics, on="topic_id")
    df_rank = df_rank.groupby(["topic_id"]).head(top_docs)

    df_top_doc = df_rank.groupby("topic_id")["doc_id"].apply(lambda x: list(x))
    top_doc_topic_dict = df_top_doc.to_dict()

    for topic in topics:
        topic.top_doc_id = top_doc_topic_dict.get(topic.topic_id, [])

    return topics
