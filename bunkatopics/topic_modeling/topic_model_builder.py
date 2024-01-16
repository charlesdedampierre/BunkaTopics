import typing as t

import pandas as pd
from sklearn.cluster import KMeans

from bunkatopics.datamodel import ConvexHullModel, Document, Term, Topic
from bunkatopics.topic_modeling.topic_name_cleaner import remove_overlapping_terms
from bunkatopics.topic_modeling.utils import specificity
from bunkatopics.visualization.convex_hull_plotter import get_convex_hull_coord


def get_topics(
    docs: t.List[Document],
    terms: t.List[Term],
    n_clusters: int = 10,
    ngrams: list = [1, 2],
    name_length: int = 15,
    top_terms_overall: int = 1000,
    min_count_terms: int = 2,
    x_column: str = "x",
    y_column: str = "y",
    custom_clustering_model=None,  # Accept a custom clustering model
) -> t.List[Topic]:
    """
    Create topics from embeddings and assign names with the top terms.

    Args:
        docs (List[Document]): List of documents with embeddings.
        terms (List[Term]): List of terms.
        n_clusters (int): Number of clusters for K-Means.
        ngrams (list): List of n-gram lengths to consider.
        name_length (int): Maximum length of topic names.
        top_terms_overall (int): Number of top terms to consider overall.
        min_count_terms (int): Minimum count of terms to consider.
        x_column (str): Column name for x-coordinate in the DataFrame.
        y_column (str): Column name for y-coordinate in the DataFrame.
        custom_clustering_model: Custom clustering model (e.g., KMeans instance).

    Returns:
        List[Topic]: List of topics with assigned names.
    """
    if custom_clustering_model is None:
        clustering_model = KMeans(n_clusters=n_clusters, n_init="auto")
    else:
        clustering_model = custom_clustering_model

    # Rest of the function remains the same...

    x_values = [getattr(doc, x_column) for doc in docs]
    y_values = [getattr(doc, y_column) for doc in docs]

    # Rest of the function remains unchanged...

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

    topic_doc_dict = df_embeddings_2D["topic_id"].to_dict()
    for doc in docs:
        doc.topic_id = topic_doc_dict.get(doc.doc_id, [])

    terms = [x for x in terms if x.count_terms >= min_count_terms]
    df_terms = pd.DataFrame.from_records([term.dict() for term in terms])
    df_terms = df_terms.sort_values("count_terms", ascending=False)
    df_terms = df_terms.head(top_terms_overall)
    df_terms = df_terms[df_terms["ngrams"].isin(ngrams)]

    df_terms_indexed = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_terms_indexed = df_terms_indexed[["doc_id", "term_id", "topic_id"]]
    df_terms_indexed = df_terms_indexed.explode("term_id").reset_index(drop=True)

    df_terms_topics = pd.merge(df_terms_indexed, df_terms, on="term_id")

    df_topics_rep = specificity(
        df_terms_topics, X="topic_id", Y="term_id", Z=None, top_n=500
    )
    df_topics_rep = (
        df_topics_rep.groupby("topic_id")["term_id"].apply(list).reset_index()
    )
    df_topics_rep["name"] = df_topics_rep["term_id"].apply(lambda x: x[:100])
    df_topics_rep["name"] = df_topics_rep["name"].apply(
        lambda x: remove_overlapping_terms(x)
    )

    df_topics_rep["name"] = df_topics_rep["name"].apply(lambda x: x[:name_length])
    df_topics_rep["name"] = df_topics_rep["name"].apply(lambda x: " | ".join(x))

    topics = [Topic(**x) for x in df_topics_rep.to_dict(orient="records")]

    df_topics_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
    df_topics_docs = df_topics_docs[["doc_id", "x", "y", "topic_id"]]
    df_topics_docs = df_topics_docs.groupby("topic_id").agg(
        size=("doc_id", "count"), x_centroid=("x", "mean"), y_centroid=("y", "mean")
    )

    topic_dict = df_topics_docs[["size", "x_centroid", "y_centroid"]].to_dict("index")

    for topic in topics:
        topic.size = topic_dict[topic.topic_id]["size"]
        topic.x_centroid = topic_dict[topic.topic_id]["x_centroid"]
        topic.y_centroid = topic_dict[topic.topic_id]["y_centroid"]

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
    except Exception as e:
        print(e)

    return topics
