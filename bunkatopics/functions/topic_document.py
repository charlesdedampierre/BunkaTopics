import typing as t

import pandas as pd

from ..datamodel import Document, Topic, TopicRanking


def get_top_documents(
    docs: t.List[Document],
    topics: t.List[Topic],
    ranking_terms=20,
) -> t.List[Document]:
    ranking_terms = 20

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

    df_rank["rank"] = df_rank.groupby("topic_id")["count_topic_terms"].rank(
        method="first", ascending=False
    )

    doc_ids = list(df_rank["doc_id"])
    topic_ids = list(df_rank["topic_id"])
    ranks = list(df_rank["rank"])

    final_dict = {}
    for doc_id, topic_id, rank in zip(doc_ids, topic_ids, ranks):
        res = TopicRanking(topic_id=topic_id, rank=rank)
        final_dict[doc_id] = res

    for doc in docs:
        doc.topic_ranking = final_dict.get(doc.doc_id)

    return docs
