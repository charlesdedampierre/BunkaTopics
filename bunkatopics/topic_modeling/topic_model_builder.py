import typing as t

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from bunkatopics.datamodel import ConvexHullModel, Document, Term, Topic
from bunkatopics.logging import logger
from bunkatopics.topic_modeling.utils import specificity
from bunkatopics.visualization.convex_hull_plotter import get_convex_hull_coord

pd.options.mode.chained_assignment = None


class BunkaTopicModelingND:
    """
    A class to perform topic modeling on a set of documents using n-dimensional embeddings.

    This class utilizes clustering (default KMeans) to identify topics within a collection of documents.
    Unlike the standard BunkaTopicModeling, this class can work with embeddings of any dimension,
    not just 2D coordinates. Topics are formed based on these n-dimensional embeddings but can still
    be visualized in 2D.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        ngrams: list = [1, 2],
        name_length: int = 15,
        top_terms_overall: int = 1000,
        min_count_terms: int = 2,
        min_docs_per_cluster: int = 10,
        n_dimensions: int = 10,
        custom_clustering_model=None,
    ) -> None:
        """Constructs all the necessary attributes for the BunkaTopicModelingND object.

        Arguments:
            n_clusters (int, optional): Number of clusters for K-Means. Defaults to 10.
            ngrams (list, optional): List of n-gram lengths to consider. Defaults to [1, 2].
            name_length (int, optional): Maximum length of topic names. Defaults to 15.
            top_terms_overall (int, optional): Number of top terms to consider overall. Defaults to 1000.
            min_count_terms (int, optional): Minimum count of terms to be considered. Defaults to 2.
            min_docs_per_cluster (int, optional): Minimum count of documents per topic. Defaults to 10.
            n_dimensions (int, optional): Number of dimensions of the embeddings. Defaults to 10.
            custom_clustering_model (optional): Custom clustering model instance, if any. Defaults to None.
        """

        self.n_clusters = n_clusters
        self.ngrams = ngrams
        self.name_length = name_length
        self.top_terms_overall = top_terms_overall
        self.min_count_terms = min_count_terms
        self.min_docs_per_cluster = min_docs_per_cluster
        self.n_dimensions = n_dimensions
        self.custom_clustering_model = custom_clustering_model

    def fit_transform(
        self,
        docs: t.List[Document],
        terms: t.List[Term],
    ) -> t.List[Topic]:
        """
        Analyzes documents and terms to form topics using n-dimensional embeddings.

        This method performs clustering on the n-dimensional document embeddings to identify distinct topics.
        Each topic is named based on the top terms associated with it. The method also calculates
        additional topic properties such as centroid coordinates in both n-dimensional space and 2D space.

        Arguments:
            docs (List[Document]): List of Document objects representing the documents to be analyzed.
            terms (List[Term]): List of Term objects representing the terms to be considered in topic naming.
        Returns:
            List[Topic]: A list of Topic objects, each representing a discovered topic with attributes
                     like name, size, centroid coordinates, and convex hull.

        Notes:
            - If a custom clustering model is not provided, the method defaults to using KMeans for clustering.
            - Topics are named using the most significant terms within each cluster.
            - The method calculates both n-dimensional centroids and 2D centroids for visualization.
            - Convex hulls are still created in 2D for visualization purposes.
        """
        logger.info(
            f"Performing topic modeling using {self.n_dimensions}-dimensional embeddings"
        )

        # Extract n-dimensional embeddings
        nd_embeddings = []
        doc_ids = []

        # Get documents with n-dimensional embeddings
        valid_docs = []
        for doc in docs:
            if (
                hasattr(doc, "nd_embedding")
                and doc.nd_embedding
                and len(doc.nd_embedding) >= self.n_dimensions
            ):
                nd_embeddings.append(doc.nd_embedding)
                doc_ids.append(doc.doc_id)
                valid_docs.append(doc)

        # Check if we have valid embeddings
        if not nd_embeddings:
            logger.warning(
                "No valid n-dimensional embeddings found. Make sure documents have nd_embedding attribute."
            )
            # Fall back to using x, y coordinates if available
            logger.info("Falling back to 2D coordinates for clustering")
            x_values = [getattr(doc, "x", None) for doc in docs]
            y_values = [getattr(doc, "y", None) for doc in docs]

            if all(x is not None for x in x_values) and all(
                y is not None for y in y_values
            ):
                # Use 2D coordinates as a fallback
                df_embeddings_2D = pd.DataFrame(
                    {
                        "doc_id": [doc.doc_id for doc in docs],
                        "x": x_values,
                        "y": y_values,
                    }
                )
                df_embeddings_2D = df_embeddings_2D.set_index("doc_id")

                # Setup clustering
                if self.custom_clustering_model is None:
                    clustering_model = KMeans(
                        n_clusters=self.n_clusters, n_init="auto", random_state=42
                    )
                else:
                    clustering_model = self.custom_clustering_model

                # Perform 2D clustering as fallback
                df_embeddings_2D["topic_number"] = clustering_model.fit(
                    df_embeddings_2D
                ).labels_.astype(str)

                df_embeddings_2D["topic_id"] = (
                    "bt" + "-" + df_embeddings_2D["topic_number"]
                )

                # For the case of HDBSCAN
                df_embeddings_2D.loc[
                    df_embeddings_2D["topic_id"] == "bt--1", "topic_id"
                ] = "bt-no-topic"

                topic_doc_dict = df_embeddings_2D["topic_id"].to_dict()
                for doc in docs:
                    doc.topic_id = topic_doc_dict.get(doc.doc_id, "bt-no-topic")
            else:
                logger.error("No valid embeddings found. Cannot perform clustering.")
                return []
        else:
            # We have n-dimensional embeddings, proceed with n-dimensional clustering
            logger.info(
                f"Clustering {len(nd_embeddings)} documents in {self.n_dimensions}D space"
            )

            # Setup clustering model
            if self.custom_clustering_model is None:
                clustering_model = KMeans(
                    n_clusters=self.n_clusters, n_init="auto", random_state=42
                )
            else:
                clustering_model = self.custom_clustering_model

            # Perform n-dimensional clustering
            nd_embeddings_array = np.array(nd_embeddings)
            cluster_labels = clustering_model.fit(nd_embeddings_array).labels_.astype(
                str
            )

            # Create a mapping from document ID to topic ID
            topic_mapping = {}
            for i, doc_id in enumerate(doc_ids):
                topic_id = "bt-" + cluster_labels[i]
                if topic_id == "bt--1":  # Handle HDBSCAN noise points
                    topic_id = "bt-no-topic"
                topic_mapping[doc_id] = topic_id

            # Assign topic IDs to all documents
            for doc in docs:
                doc.topic_id = topic_mapping.get(doc.doc_id, "bt-no-topic")

        # Process terms for naming topics - same as in original implementation
        terms = [x for x in terms if x.count_terms >= self.min_count_terms]

        df_terms = pd.DataFrame.from_records([term.model_dump() for term in terms])
        if df_terms.empty:
            logger.warning(
                "No terms found with sufficient count. Consider lowering min_count_terms."
            )
            return []

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
        df_topics_rep.loc[df_topics_rep["topic_id"] == "bt-no-topic", "name"] = (
            "no-topic"
        )

        # Create Topic objects
        topics = [Topic(**x) for x in df_topics_rep.to_dict(orient="records")]

        # Calculate both 2D and n-dimensional centroids
        # First, the 2D centroids for visualization
        df_topics_docs = pd.DataFrame.from_records([doc.model_dump() for doc in docs])
        df_topics_docs = df_topics_docs[["doc_id", "x", "y", "topic_id"]]
        df_topics_docs = df_topics_docs.groupby("topic_id").agg(
            size=("doc_id", "count"), x_centroid=("x", "mean"), y_centroid=("y", "mean")
        )

        topic_dict = df_topics_docs[["size", "x_centroid", "y_centroid"]].to_dict(
            "index"
        )

        # Next, calculate n-dimensional centroids
        nd_centroids = {}
        for topic_id in set(
            doc.topic_id for doc in docs if doc.topic_id != "bt-no-topic"
        ):
            # Get documents in this topic that have n-dimensional embeddings
            topic_docs_with_nd = [
                doc
                for doc in docs
                if doc.topic_id == topic_id
                and hasattr(doc, "nd_embedding")
                and doc.nd_embedding
            ]

            if topic_docs_with_nd:
                # Calculate n-dimensional centroid
                nd_centroid = np.mean(
                    [doc.nd_embedding for doc in topic_docs_with_nd], axis=0
                ).tolist()
                nd_centroids[topic_id] = nd_centroid

        # Update Topic objects with size and centroid information
        for topic in topics:
            if topic.topic_id in topic_dict:
                topic.size = topic_dict[topic.topic_id]["size"]
                topic.x_centroid = topic_dict[topic.topic_id]["x_centroid"]
                topic.y_centroid = topic_dict[topic.topic_id]["y_centroid"]

                # Add n-dimensional centroid if available
                if topic.topic_id in nd_centroids:
                    topic.nd_centroid = nd_centroids[topic.topic_id]

        # Remove topics with too few documents
        topics = [
            x for x in topics if getattr(x, "size", 0) >= self.min_docs_per_cluster
        ]

        return topics


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
        min_docs_per_cluster: int = 10,
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
            min_docs_per_cluster (int, optional): Minimum count of documents per topic
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
        self.min_docs_per_cluster = min_docs_per_cluster

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

        if self.custom_clustering_model is None:
            clustering_model = KMeans(
                n_clusters=self.n_clusters, n_init="auto", random_state=42
            )

        else:
            clustering_model = self.custom_clustering_model

        df_embeddings_2D["topic_number"] = clustering_model.fit(
            df_embeddings_2D
        ).labels_.astype(str)

        df_embeddings_2D["topic_id"] = "bt" + "-" + df_embeddings_2D["topic_number"]

        # For the case of HDBSCAN
        df_embeddings_2D["topic_id"][
            df_embeddings_2D["topic_id"] == "bt--1"
        ] = "bt-no-topic"

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
        df_topics_rep["name"][df_topics_rep["topic_id"] == "bt-no-topic"] = "no-topic"

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

        # remove too small clusters
        topics = [x for x in topics if x.size >= self.min_docs_per_cluster]
        try:
            for x in topics:
                topic_id = x.topic_id
                if topic_id != "bt-no-topic":
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

        # Remove in case of HDBSCAN ?
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
