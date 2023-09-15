from bunkatopics.datamodel import DOC_ID
import typing as t
from sklearn.cluster import KMeans
import pandas as pd
from bunkatopics.functions.utils import specificity
from bunkatopics.datamodel import Topic, Document


def topic_modeling(
    dict_doc_embeddings: t.Dict[DOC_ID, t.Dict[str, int]],
    dict_doc_terms: t.Dict[DOC_ID, t.Dict[str, int]],
    n_clusters=10,
):
    clustering_model = KMeans(n_clusters=n_clusters)

    df_embeddings_2D = pd.DataFrame(dict_doc_embeddings).T
    df_embeddings_2D["topic_number"] = clustering_model.fit(
        df_embeddings_2D
    ).labels_.astype(str)
    df_embeddings_2D["topic_id"] = "bt" + "-" + df_embeddings_2D["topic_number"]
    df_embeddings_2D = df_embeddings_2D.reset_index()
    df_embeddings_2D = df_embeddings_2D.rename(columns={"index": "doc_id"})

    df_topics_docs = df_embeddings_2D.groupby("topic_id").agg(
        size=("doc_id", "count"), x_centroid=("x", "mean"), y_centroid=("y", "mean")
    )
    df_topics_docs = df_topics_docs.reset_index()

    df_terms_topics = pd.DataFrame(dict_doc_terms).T.reset_index()
    df_terms_topics = df_terms_topics.rename(columns={"index": "doc_id"})
    df_terms_topics = pd.merge(df_terms_topics, df_embeddings_2D, on="doc_id")
    df_terms_topics = df_terms_topics.explode("term_id").reset_index(drop=True)

    df_topics_rep = specificity(
        df_terms_topics, X="topic_id", Y="term_id", Z=None, top_n=500
    )

    df_topics_rep = (
        df_topics_rep.groupby("topic_id")["term_id"].apply(list).reset_index()
    )

    df_topics = pd.merge(df_topics_rep, df_topics_docs, on="topic_id")
    df_topics = df_topics.set_index("topic_id")

    df_embeddings_2D = df_embeddings_2D.set_index("doc_id")

    dict_topics = df_topics.to_dict(orient="index")
    dict_docs = df_embeddings_2D.to_dict(orient="index")

    return dict_topics, dict_docs
