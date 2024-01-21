import typing as t

import pandas as pd

from bunkatopics.datamodel import Document, Topic, TopicRanking


class DocumentRanker:
    def __init__(self, ranking_terms: int = 20) -> None:
        self.ranking_terms = ranking_terms

    def fit_transform(
        self,
        docs: t.List[Document],
        topics: t.List[Topic],
    ) -> t.Tuple[t.List[Document], t.List[Topic]]:
        """
        Calculate top documents for each topic based on ranking terms.

        Args:
            docs (List[Document]): List of documents.
            topics (List[Topic]): List of topics.
            ranking_terms (int): Number of topic specific terms to hekp ranking the topics

        Returns:
            Tuple[List[Document], List[Topic]]: Updated lists of documents and topics.
        """
        # Create a DataFrame from the list of documents
        df_docs = pd.DataFrame.from_records([doc.model_dump() for doc in docs])

        # Explode the term_id column to have one row per term
        df_docs = df_docs[["doc_id", "topic_id", "term_id"]]
        df_docs = df_docs.explode("term_id").reset_index(drop=True)

        # Create a DataFrame from the list of topics and truncate term_id
        df_topics = pd.DataFrame.from_records([topic.model_dump() for topic in topics])
        df_topics["term_id"] = df_topics["term_id"].apply(
            lambda x: x[: self.ranking_terms]
        )
        df_topics = df_topics[["topic_id", "term_id"]]
        df_topics = df_topics.explode("term_id").reset_index(drop=True)

        # Merge documents and topics, and calculate term counts
        df_rank = pd.merge(df_docs, df_topics, on=["topic_id", "term_id"])
        df_rank = (
            df_rank.groupby(["topic_id", "doc_id"])["term_id"]
            .count()
            .rename("count_topic_terms")
            .reset_index()
        )

        # Sort and rank documents within each topic
        df_rank = df_rank.sort_values(
            ["topic_id", "count_topic_terms"], ascending=(True, False)
        ).reset_index(drop=True)
        df_rank["rank"] = df_rank.groupby("topic_id")["count_topic_terms"].rank(
            method="first", ascending=False
        )

        # Create a dictionary of TopicRanking objects for each document
        final_dict = {}
        for doc_id, topic_id, rank in zip(
            df_rank["doc_id"], df_rank["topic_id"], df_rank["rank"]
        ):
            res = TopicRanking(topic_id=topic_id, rank=rank)
            final_dict[doc_id] = res

        # Update each document with its topic ranking
        for doc in docs:
            doc.topic_ranking = final_dict.get(doc.doc_id)

        # Create a DataFrame for document content
        df_content = pd.DataFrame.from_records([doc.model_dump() for doc in docs])
        df_content = df_content[["doc_id", "content"]]

        # Merge document content with topic information
        df_topics_rank = pd.merge(df_rank, df_content, on="doc_id")
        df_topics_rank = df_topics_rank[["topic_id", "content"]]
        df_topics_rank = df_topics_rank.groupby("topic_id")["content"].apply(list)

        # Create a dictionary of top document content for each topic
        dict_topic_rank = df_topics_rank.to_dict()

        # Update each topic with its top document content
        for topic in topics:
            topic.top_doc_content = dict_topic_rank.get(topic.topic_id)

        return docs, topics
