import typing as t

import pandas as pd
from sklearn.cluster import KMeans

from bunkatopics.datamodel import ConvexHullModel, Document, Term, Topic
from bunkatopics.topic_modeling.utils import specificity
from bunkatopics.visualization.convex_hull_plotter import get_convex_hull_coord


class BunkaTopicModeling:
    """
    A class to perform topic modeling on a set of documents.

    This class utilizes clustering (default KMeans) to identify topics within a collection of documents.
    Each document and term is represented by embeddings, and topics are formed based on these embeddings.
    Topics are named using the top terms associated with them."""

    def __init__(
        self,
        n_clusters: int = 10,
        ngrams: list = [1, 2],
        name_length: int = 15,
        top_terms_overall: int = 1000,
        min_count_terms: int = 2,
        x_column: str = "x",
        y_column: str = "y",
        custom_clustering_model=None,
    ) -> None:
        """Constructs all the necessary attributes for the BunkaTopicModeling object.

        Arguments:
            n_clusters (int, optional): Number of clusters for K-Means. Defaults to 10.
            ngrams (list, optional): List of n-gram lengths to consider. Defaults to [1, 2].
            name_length (int, optional): Maximum length of topic names. Defaults to 15.
            top_terms_overall (int, optional): Number of top terms to consider overall. Defaults to 1000.
            min_count_terms (int, optional): Minimum count of terms to be considered. Defaults to 2.
            x_column (str, optional): Column name for x-coordinate in the DataFrame. Defaults to "x".
            y_column (str, optional): Column name for y-coordinate in the DataFrame. Defaults to "y".
            custom_clustering_model (optional): Custom clustering model instance, if any. Defaults to None.
        """

        self.n_clusters = n_clusters
        self.ngrams = ngrams
        self.name_length = name_length
        self.top_terms_overall = top_terms_overall
        self.min_count_terms = min_count_terms
        self.x_column = x_column
        self.y_column = y_column
        self.custom_clustering_model = custom_clustering_model

    def fit_transform(
        self,
        docs: t.List[Document],
        terms: t.List[Term],
    ) -> t.List[Topic]:
        """
        Analyzes documents and terms to form topics, assigns names to these topics based on the top terms,
        and returns a list of Topic instances.

        This method performs clustering on the document embeddings to identify distinct topics.
        Each topic is named based on the top terms associated with it. The method also calculates
        additional topic properties such as centroid coordinates and convex hulls.

        Arguments:
            docs (List[[Document]): List of Document objects representing the documents to be analyzed.
            terms (List[Term]): List of Term objects representing the terms to be considered in topic naming.
        Returns:
            List[Topic]: A list of Topic objects, each representing a discovered topic with attributes
                     like name, size, centroid coordinates, and convex hull.

        Notes:
            - If a custom clustering model is not provided, the method defaults to using KMeans for clustering.
            - Topics are named using the most significant terms within each cluster.
            - The method calculates the centroid and convex hull for each topic based on the document embeddings.
        """
        if self.custom_clustering_model is None:
            clustering_model = KMeans(n_clusters=self.n_clusters, n_init="auto")
        else:
            clustering_model = self.custom_clustering_model

        # Rest of the function remains the same...

        x_values = [getattr(doc, self.x_column) for doc in docs]
        y_values = [getattr(doc, self.y_column) for doc in docs]

        # Rest of the function remains unchanged...

        df_embeddings_2D = pd.DataFrame(
            {
                "doc_id": [doc.doc_id for doc in docs],
                self.x_column: x_values,
                self.y_column: y_values,
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

        terms = [x for x in terms if x.count_terms >= self.min_count_terms]
        df_terms = pd.DataFrame.from_records([term.model_dump() for term in terms])
        df_terms = df_terms.sort_values("count_terms", ascending=False)
        df_terms = df_terms.head(self.top_terms_overall)
        df_terms = df_terms[df_terms["ngrams"].isin(self.ngrams)]

        df_terms_indexed = pd.DataFrame.from_records([doc.model_dump() for doc in docs])
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
        df_topics_rep["name"] = df_topics_rep["name"].apply(lambda x: clean_terms(x))

        df_topics_rep["name"] = df_topics_rep["name"].apply(
            lambda x: x[: self.name_length]
        )
        df_topics_rep["name"] = df_topics_rep["name"].apply(lambda x: " | ".join(x))

        topics = [Topic(**x) for x in df_topics_rep.to_dict(orient="records")]

        df_topics_docs = pd.DataFrame.from_records([doc.model_dump() for doc in docs])
        df_topics_docs = df_topics_docs[["doc_id", "x", "y", "topic_id"]]
        df_topics_docs = df_topics_docs.groupby("topic_id").agg(
            size=("doc_id", "count"), x_centroid=("x", "mean"), y_centroid=("y", "mean")
        )

        topic_dict = df_topics_docs[["size", "x_centroid", "y_centroid"]].to_dict(
            "index"
        )

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


def clean_terms(terms: t.List[str]) -> t.List[str]:
    """
    Remove overlapping terms from a list of terms.

    Args:
        terms (List[str]): List of terms to process.

    Returns:
        List[str]: List of terms with overlapping terms removed.
    """
    seen_words = set()
    filtered_terms = []

    for term in terms:
        # Remove leading and trailing spaces and convert to lowercase
        cleaned_term = term.strip()

        # Skip the term 'CUR'
        if cleaned_term == "CUR":
            continue

        # Skip terms with one letter or number or with only alpha-numeric sign
        if (
            len(cleaned_term) <= 1
            or cleaned_term.isnumeric()
            or not cleaned_term.isalpha()
        ):
            continue

        # Check if the cleaned term consists of only alphabetical characters
        if all(char.isalpha() for char in cleaned_term):
            # Check if the cleaned term is in the seen_words set
            if cleaned_term not in seen_words:
                filtered_terms.append(cleaned_term)
                seen_words.add(cleaned_term)

    # Create a dictionary to store terms with lowercase keys
    term_dict = {}

    for term in filtered_terms:
        # Convert the term to lowercase to use as the key
        lowercase_term = term.lower()

        # Check if the lowercase term is not already in the dictionary
        # If it's not in the dictionary or if the original term is uppercase, add it
        if lowercase_term not in term_dict or term.isupper():
            term_dict[lowercase_term] = term

    # Extract the unique terms (case-insensitive) from the dictionary values
    result = list(term_dict.values())

    return result
