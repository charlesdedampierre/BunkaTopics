import sys

sys.path.append("../")
import typing as t

import pandas as pd
from sklearn.cluster import KMeans

from bunkatopics.datamodel import (
    DOC_ID,
    TERM_ID,
    TOPIC_ID,
    ConvexHullModel,
    Document,
    Term,
    Topic,
)
from bunkatopics.functions.topic_representation import remove_overlapping_terms
from bunkatopics.functions.utils import specificity
from bunkatopics.visualisation.convex_hull import get_convex_hull_coord


def get_topics(
    docs: t.List[Document],
    terms: t.List[Term],
    n_clusters: int = 10,
    ngrams: list = [1, 2],
    name_lenght: int = 15,
    top_terms_overall: int = 1000,
    x_column="x",
    y_column="y",
) -> t.List[Topic]:
    """Create Topics from an embeddings and give names with the top terms"""
    clustering_model = KMeans(n_clusters=n_clusters)

    x_values = [getattr(doc, x_column) for doc in docs]
    y_values = [getattr(doc, y_column) for doc in docs]

    df_embeddings_2D = pd.DataFrame(
        {
            "doc_id": [doc.doc_id for doc in docs],
            x_column: x_values,
            y_column: y_values,
        }
    )
    df_embeddings_2D = df_embeddings_2D.set_index("doc_id")

    df_embeddings_2D["topic_number"] = clustering_model.fit(
        df_embeddings_2D
    ).labels_.astype(str)

    df_embeddings_2D["topic_id"] = "bt" + "-" + df_embeddings_2D["topic_number"]

    # insert into the documents
    topic_doc_dict = df_embeddings_2D["topic_id"].to_dict()
    for doc in docs:
        doc.topic_id = topic_doc_dict.get(doc.doc_id, [])

    df_terms = pd.DataFrame.from_records([term.dict() for term in terms])
    df_terms = df_terms.sort_values("count_terms", ascending=False)
    df_terms = df_terms.head(top_terms_overall)
    df_terms = df_terms[df_terms["ngrams"].isin(ngrams)]

    df_terms_indexed = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_terms_indexed = df_terms_indexed[["doc_id", "term_id", "topic_id"]]
    df_terms_indexed = df_terms_indexed.explode("term_id").reset_index(drop=True)

    df_terms_topics = pd.merge(df_terms, df_terms_indexed, on="term_id")

    terms_type = "term_id"
    df_topics_rep = specificity(
        df_terms_topics, X="topic_id", Y=terms_type, Z=None, top_n=500
    )
    df_topics_rep = (
        df_topics_rep.groupby("topic_id")["term_id"].apply(list).reset_index()
    )
    df_topics_rep["name"] = df_topics_rep["term_id"].apply(lambda x: x[:100])
    df_topics_rep["name"] = df_topics_rep["name"].apply(
        lambda x: remove_overlapping_terms(x)
    )

    df_topics_rep["name"] = df_topics_rep["name"].apply(lambda x: x[:name_lenght])
    df_topics_rep["name"] = df_topics_rep["name"].apply(lambda x: " | ".join(x))

    topics = [Topic(**x) for x in df_topics_rep.to_dict(orient="records")]

    df_topics_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_topics_docs = df_topics_docs[["doc_id", "x", "y", "topic_id"]]
    df_topics_docs = df_topics_docs.groupby("topic_id").agg(
        size=("doc_id", "count"), x_centroid=("x", "mean"), y_centroid=("y", "mean")
    )

    topic_dict = df_topics_docs[["size", "x_centroid", "y_centroid"]].to_dict("index")

    # Update the documents with the x and y values from the DataFrame
    for topic in topics:
        topic.size = topic_dict[topic.topic_id]["size"]
        topic.x_centroid = topic_dict[topic.topic_id]["x_centroid"]
        topic.y_centroid = topic_dict[topic.topic_id]["y_centroid"]

    # Compute Convex Hull
    try:
        for x in topics:
            topic_id = x.topic_id
            x_points = [doc.x for doc in docs if doc.topic_id == topic_id]
            y_points = [doc.y for doc in docs if doc.topic_id == topic_id]

            points = pd.DataFrame({"x": x_points, "y": y_points}).values

            x_ch, y_ch = get_convex_hull_coord(points, interpolate_curve=True)
            x_ch = list(x_ch)
            y_ch = list(y_ch)

            res = ConvexHullModel(x_coordinates=x_ch, y_coordinates=y_ch)
            x.convex_hull = res
    except:
        pass
    # df_topics = pd.DataFrame.from_records([topic.dict() for topic in topics])
    return topics
