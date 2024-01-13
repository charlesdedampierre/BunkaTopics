from typing import List
import pandas as pd
from sklearn.cluster import KMeans
from bunkatopics.datamodel import Document, Term, Topic, ConvexHullModel
from bunkatopics.functions.topic_representation import remove_overlapping_terms
from bunkatopics.functions.utils import specificity
from bunkatopics.visualisation.convex_hull import get_convex_hull_coord


def get_topics(
    docs: List[Document],
    terms: List[Term],
    n_clusters: int = 10,
    ngrams: List[int] = [1, 2],
    name_length: int = 15,
    top_terms_overall: int = 1000,
    min_count_terms: int = 20,
    x_column: str = "x",
    y_column: str = "y",
) -> List[Topic]:
    """
    Create topics from embeddings and assign names with the top terms.

    Args:
        docs (List[Document]): List of documents.
        terms (List[Term]): List of terms.
        n_clusters (int, optional): The number of clusters to generate. Default is 10.
        ngrams (List[int], optional): The ngrams to use for clustering. Default is [1, 2].
        name_length (int, optional): The maximum length of the generated topic names. Default is 15.
        top_terms_overall (int, optional): The number of top terms to use overall for clustering. Default is 1000.
        min_count_terms (int, optional): The minimum document count for terms to be included. Default is 20.
        x_column (str, optional): Name of the x-coordinate column. Default is "x".
        y_column (str, optional): Name of the y-coordinate column. Default is "y".

    Returns:
        List[Topic]: List of generated topics.
    """
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

    # Insert into the documents
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

    df_topics_rep["name"] = df_topics_rep["name"].apply(lambda x: x[:name_length])
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
        for topic in topics:
            topic_id = topic.topic_id
            x_points = [doc.x for doc in docs if doc.topic_id == topic_id]
            y_points = [doc.y for doc in docs if doc.topic_id == topic_id]

            points = pd.DataFrame({"x": x_points, "y": y_points}).values

            x_ch, y_ch = get_convex_hull_coord(points, interpolate_curve=True)
            x_ch = list(x_ch)
            y_ch = list(y_ch)

            res = ConvexHullModel(x_coordinates=x_ch, y_coordinates=y_ch)
            topic.convex_hull = res
    except Exception:
        pass

    return topics
