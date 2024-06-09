import copy
import json
import os
import random
import subprocess
import typing as t
import uuid
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from FlagEmbedding import FlagModel
from IPython.display import display
from ipywidgets import Button, Checkbox, Label, Layout, VBox, widgets
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from numba.core.errors import NumbaDeprecationWarning
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from bunkatopics.bourdieu import (
    BourdieuAPI,
    BourdieuOneDimensionVisualizer,
    BourdieuVisualizer,
)
from bunkatopics.datamodel import (
    DOC_ID,
    BourdieuQuery,
    Document,
    Topic,
    TopicGenParam,
    TopicParam,
)
from bunkatopics.logging import logger
from bunkatopics.serveur import is_server_running, kill_server
from bunkatopics.topic_modeling import (
    BunkaTopicModeling,
    DocumentRanker,
    LLMCleaningTopic,
    TextacyTermsExtractor,
)
from bunkatopics.topic_modeling.topic_utils import get_topic_repartition
from bunkatopics.topic_modeling.utils import (
    detect_language,
    detect_language_to_language_name,
)
from bunkatopics.utils import (
    BunkaError,
    _create_topic_dfs,
    _filter_hdbscan,
    count_tokens,
)
from bunkatopics.visualization import TopicVisualizer
from bunkatopics.visualization.query_visualizer import plot_query

# Filter ResourceWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="omp_set_nested routine deprecated")


os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Bunka:
    """The Bunka class for managing and analyzing textual data using various NLP techniques.

    Examples:
    ```python
    from bunkatopics import Bunka
    from datasets import load_dataset
    import random

    # Extract Data
    dataset = load_dataset("rguo123/trump_tweets")["train"]["content"]
    docs = random.sample(dataset, 1000)

    bunka = Bunka()
    topics = bunka.fit_transform(docs)
    bunka.visualize_topics(width=800, height=800)
    ```
    """

    def __init__(
        self,
        embedding_model: Embeddings = None,
        projection_model=None,
        language: str = "english",  # will be removed in the future
    ):
        """Initialize a BunkaTopics instance.

        Args:
            embedding_model (Embeddings, optional): An optional embedding model for generating document embeddings.
                If not provided, a default model will be used based on the specified language.
                Default is None.
            projection_model (optional): An optional projection model to reduce the dimensionality of the embeddings.
                Default is None.
        """
        warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
        if embedding_model is None:
            embedding_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")

        if projection_model is None:
            projection_model = umap.UMAP(
                n_components=2,
                random_state=42,
            )

        self.projection_model = projection_model
        self.embedding_model = embedding_model
        self.df_cleaned = None

    def fit(
        self,
        docs: t.List[str],
        ids: t.List[DOC_ID] = None,
        pre_computed_embeddings: t.Optional[
            t.List[t.Dict[DOC_ID, t.List[float]]]
        ] = None,
        metadata: t.Optional[t.List[dict]] = None,
        sampling_size_for_terms: t.Optional[int] = 1000,
        language: bool = None,
    ) -> None:
        """
        Fits the Bunka model to the provided list of documents.

        This method processes the documents, extracts terms, generates embeddings, and
        applies dimensionality reduction to prepare the data for topic modeling.

        Args:
            docs (t.List[str]): A list of document strings.
            ids (t.Optional[t.List[DOC_ID]]): Optional. A list of identifiers for the documents. If not provided, UUIDs are generated.
            metadata (t.Optional[t.List[str]): A of metadata dictionaries for the documents.
            sampling_size_for_terms (t.Optional[int]): The number of documents to sample for term extraction. Default is 2000.
        """

        df = pd.DataFrame(docs, columns=["content"])

        # Transform into a Document model
        if ids is not None:
            ids = [str(x) for x in ids]
            df["doc_id"] = ids
            df = df.drop_duplicates(subset="doc_id", keep="first")

        else:
            df["doc_id"] = [str(uuid.uuid4())[:20] for _ in range(len(df))]

        if metadata is not None:
            metadata_values = [
                {key: metadata[key][i] for key in metadata} for i in range(len(df))
            ]

            df["metadata"] = metadata_values

        df = df[~df["content"].isna()]
        df = df.reset_index(drop=True)

        self.docs = [Document(**row) for row in df.to_dict(orient="records")]
        sentences = [doc.content for doc in self.docs]

        total_number_of_tokens = count_tokens(sentences)
        logger.info(f"Processing {total_number_of_tokens} tokens")

        ids = [doc.doc_id for doc in self.docs]

        # Detect language

        sample_size = len(sentences) // 100  # sample 1% of the dataset

        # Randomly sample 1% of the dataset
        sampled_sentences = random.sample(sentences, sample_size)

        if language is None:
            self.detected_language = detect_language(sampled_sentences)
        else:
            self.detected_language = language
        self.language_name = detect_language_to_language_name.get(
            self.detected_language, "english"
        )

        logger.info(f"Detected language: {self.language_name}")

        # if self.language_name != "english":
        #     embedding_model = SentenceTransformer(
        #         model_name_or_path="paraphrase-multilingual-MiniLM-L12-v2"
        #     )

        logger.info(
            "Embedding documents... (can take varying amounts of time depending on their size)"
        )

        if pre_computed_embeddings is None:
            # Determine if self.embedding_model is an instance of SentenceTransformer
            if isinstance(self.embedding_model, SentenceTransformer):
                bunka_embeddings = self.embedding_model.encode(
                    sentences, show_progress_bar=True
                )
                bunka_embeddings = bunka_embeddings.tolist()

            elif isinstance(self.embedding_model, HuggingFaceEmbeddings):
                bunka_embeddings = self.embedding_model.embed_documents(sentences)

            elif isinstance(self.embedding_model, FlagModel):
                bunka_embeddings = self.embedding_model.encode(sentences)
                bunka_embeddings = bunka_embeddings.tolist()

            else:
                bunka_embeddings = self.embedding_model.encode(
                    sentences
                )  # show_progress_bar=True
        else:
            # pre_computed_embeddings.sort(key=lambda x: ids.index(x["doc_id"]))
            # bunka_embeddings = [x["embedding"] for x in pre_computed_embeddings]
            bunka_embeddings = []
            for x in pre_computed_embeddings:
                embedding = x["embedding"]
                if isinstance(embedding, list):
                    bunka_embeddings.append(embedding)
                else:
                    bunka_embeddings.append(embedding.tolist())

        # Add to the bunka objects
        emb_doc_dict = {x: y for x, y in zip(ids, bunka_embeddings)}
        for doc in self.docs:
            doc.embedding = emb_doc_dict.get(doc.doc_id, [])

        # Add to the bunka objects
        emb_doc_dict = {x: y for x, y in zip(ids, bunka_embeddings)}
        for doc in self.docs:
            doc.embedding = emb_doc_dict.get(doc.doc_id, [])

        # REDUCTION OF DIMENSIONS
        logger.info("Reducing the dimensions of embeddings...")

        bunka_embeddings_2D = self.projection_model.fit_transform(
            np.array(bunka_embeddings)
        )

        # Insert to the Pydantic object
        df_embeddings_2D = pd.DataFrame(bunka_embeddings_2D, columns=["x", "y"])

        df_embeddings_2D["doc_id"] = ids
        df_embeddings_2D["bunka_docs"] = sentences

        xy_dict = df_embeddings_2D.set_index("doc_id")[["x", "y"]].to_dict("index")

        # Update the documents with the x and y values from the DataFrame
        for doc in self.docs:
            doc.x = xy_dict[doc.doc_id]["x"]
            doc.y = xy_dict[doc.doc_id]["y"]

        # CREATE A PLOT

        self.fig_embeddings = self._quick_plot(df_embeddings_2D)

        logger.info("Extracting meaningful terms from documents...")
        terms_extractor = TextacyTermsExtractor(language=self.detected_language)

        if len(sentences) >= sampling_size_for_terms:
            # Pair sentences with their corresponding ids
            paired_data = list(zip(sentences, ids))
            random.seed(42)
            sampled_data = random.sample(paired_data, sampling_size_for_terms)

            # Unpack the sampled pairs back into sentences and ids lists
            sampled_sentences, sampled_ids = zip(*sampled_data)
            logger.info(
                f"Sampling {sampling_size_for_terms} documents for term extraction"
            )
            self.terms, indexed_terms_dict = terms_extractor.fit_transform(
                sampled_ids, sampled_sentences
            )

        else:
            self.terms, indexed_terms_dict = terms_extractor.fit_transform(
                ids, sentences
            )

        # add to the docs object
        for doc in self.docs:
            doc.term_id = indexed_terms_dict.get(doc.doc_id, [])

        self.topics = None

    def remove_outliers(self, threshold=6):
        """
        Removes outliers from the dataset based on a specified threshold.

        This method applies an outlier detection algorithm to identify and remove outliers
        Args:
            threshold (int): The threshold value for outlier detection. Default is 6.
        """

        from bunkatopics.cleaning.outlier_detection import remove_outliers

        cleaned_docs = remove_outliers(docs=self.docs, threshold=threshold)
        # Calculate the number of removed documents

        removed_docs_count = len(self.docs) - len(cleaned_docs)
        logger.info("Number of removed documents: {}".format(removed_docs_count))
        self.docs = cleaned_docs

    def save_bunka(self, path: str = "bunka_dumps"):
        """
        Save the Bunka model to disk.

        This method saves the Bunka model to disk by serializing its documents and terms.

        Args:
            path (str, optional): The directory path where the model will be saved.
                Defaults to "bunka_dumps".

        Examples:
        ```python
        from bunkatopics import Bunka
        bunka = Bunka()
        ...
        bunka.save_bunka('bunka_dumps')```

        """
        from .utils import save_bunka_models

        save_bunka_models(path=path, bunka=self)

    def load_bunka(self, path):
        """
        Load the Bunka model from disk.

        This method loads the Bunka model from disk by reading the serialized documents and terms.

        Args:
            path (str): The directory path from where the model will be loaded.

        Returns:
            bunka (Bunka): The loaded Bunka model.
        """
        from .utils import read_documents_from_jsonl, read_terms_from_jsonl

        documents = read_documents_from_jsonl(path + "/bunka_docs.jsonl")
        terms = read_terms_from_jsonl(path + "/bunka_terms.jsonl")

        self.docs = documents
        self.terms = terms

        return self

    def get_topics(
        self,
        n_clusters: int = 5,
        ngrams: t.List[int] = [1, 2],
        name_length: int = 5,
        top_terms_overall: int = 2000,
        min_count_terms: int = 2,
        ranking_terms: int = 20,
        max_doc_per_topic: int = 20,
        custom_clustering_model: bool = None,
        min_docs_per_cluster: int = 10,
    ) -> pd.DataFrame:
        """
        Computes and organizes topics from the documents using specified parameters.

        This method uses a topic modeling process to identify and characterize topics within the data.

        Args:
            n_clusters (int): The number of clusters to form. Default is 5.
            ngrams (t.List[int]): The n-gram range to consider for topic extraction. Default is [1, 2].
            name_length (int): The length of the name for topics. Default is 10.
            top_terms_overall (int): The number of top terms to consider overall. Default is 2000.
            min_count_terms (int): The minimum count of terms to be considered. Default is 2.
            min_docs_per_cluster (int, optional): Minimum count of documents per topic

        Returns:
            pd.DataFrame: A DataFrame containing the topics and their associated data.

        Note:
            The method applies topic modeling using the specified parameters and updates the internal state
            with the resulting topics. It also associates the identified topics with the documents.
        """

        # Add the conditional check for min_count_terms and len(self.docs)
        if min_count_terms > 1 and len(self.docs) <= 500:
            logger.info(
                f"There is not enough data to select terms with a minimum occurrence of {min_count_terms}. Setting min_count_terms to 1"
            )
            min_count_terms = 1

        logger.info("Computing the topics")

        topic_model = BunkaTopicModeling(
            n_clusters=n_clusters,
            ngrams=ngrams,
            name_length=name_length,
            x_column="x",
            y_column="y",
            top_terms_overall=top_terms_overall,
            min_count_terms=min_count_terms,
            custom_clustering_model=custom_clustering_model,
            min_docs_per_cluster=min_docs_per_cluster,
        )

        self.topics: t.List[Topic] = topic_model.fit_transform(
            docs=self.docs,
            terms=self.terms,
        )

        model_ranker = DocumentRanker(
            ranking_terms=ranking_terms, max_doc_per_topic=max_doc_per_topic
        )
        self.docs, self.topics = model_ranker.fit_transform(self.docs, self.topics)

        (
            self.topics,
            self.docs,
        ) = _filter_hdbscan(self.topics, self.docs)

        self.df_topics_, self.df_top_docs_per_topic_ = _create_topic_dfs(
            self.topics, self.docs
        )

        return self.df_topics_

    def get_clean_topic_name(
        self,
        llm: LLM,
        use_doc: bool = False,
        context: str = "everything",
    ) -> pd.DataFrame:
        """
        Enhances topic names using a language model for cleaner and more meaningful representations.

        Args:
            llm: The language model used for cleaning topic names.
            use_doc (bool): Flag to determine whether to use document context in the cleaning process. Default is False.
            context (str): The broader context within which the topics are related Default is "everything". For instance, if you are looking at Computer Science, then update context = 'Computer Science'

        Returns:
            pd.DataFrame: A DataFrame containing the topics with cleaned names.

        Note:
            This method leverages a language model to refine the names of the topics generated by the model,
            aiming for more understandable and relevant topic descriptors.
        """

        logger.info("Using LLM to make topic names cleaner")

        model_cleaning = LLMCleaningTopic(
            llm,
            language=self.language_name,
            use_doc=use_doc,
            context=context,
        )
        self.topics: t.List[Topic] = model_cleaning.fit_transform(
            self.topics,
            self.docs,
        )

        self.df_topics_, self.df_top_docs_per_topic_ = _create_topic_dfs(
            self.topics, self.docs
        )

        return self.df_topics_

    def visualize_topics(
        self,
        show_text: bool = True,
        label_size_ratio: int = 100,
        point_size_ratio: int = 100,
        width: int = 1000,
        height: int = 1000,
        colorscale: str = "delta",
        density: bool = True,
        convex_hull: bool = True,
        color: str = None,
        # search: str = None,
    ) -> go.Figure:
        """
        Generates a visualization of the identified topics in the document set.

        Args:
            show_text (bool): Whether to display text labels on the visualization. Default is True.
            label_size_ratio (int): The size ratio of the labels in the visualization. Default is 100.
            width (int): The width of the visualization figure. Default is 1000.
            height (int): The height of the visualization figure. Default is 1000.
            colorscale (str): colorscale for the Density Plot (Default is delta)
            density (bool): Whether to display a density map
            convex_hull (bool): Whether to display lines around the clusters
            color (str): What category to use to display the color

        Returns:
            go.Figure: A Plotly graph object figure representing the topic visualization.

        Note:
            This method creates a 'Bunka Map', a graphical representation of the topics,
            using Plotly for interactive visualization. It displays how documents are grouped
            into topics and can include text labels for clarity.
        """
        logger.info("Creating the Bunka Map")

        model_visualizer = TopicVisualizer(
            width=width,
            height=height,
            show_text=show_text,
            label_size_ratio=label_size_ratio,
            point_size_ratio=point_size_ratio,
            colorscale=colorscale,
            density=density,
            convex_hull=convex_hull,
        )
        fig = model_visualizer.fit_transform(self.docs, self.topics, color=color)

        return fig

    def visualize_bourdieu(
        self,
        llm: t.Optional[LLM] = None,
        x_left_words: t.List[str] = ["war"],
        x_right_words: t.List[str] = ["peace"],
        y_top_words: t.List[str] = ["men"],
        y_bottom_words: t.List[str] = ["women"],
        height: int = 1500,
        width: int = 1500,
        display_percent: bool = True,
        clustering: bool = False,
        topic_n_clusters: int = 10,
        topic_terms: int = 2,
        topic_ngrams: t.List[int] = [1, 2],
        topic_top_terms_overall: int = 1000,
        gen_topic_language: str = "english",
        manual_axis_name: t.Optional[dict] = None,
        use_doc_gen_topic: bool = False,
        radius_size: float = 0.3,
        convex_hull: bool = True,
        density: bool = True,
        colorscale: str = "delta",
        label_size_ratio_clusters: int = 100,
        label_size_ratio_label: int = 50,
        label_size_ratio_percent: int = 10,
        min_docs_per_cluster: int = 5,
    ) -> go.Figure:
        """
        Creates and visualizes a Bourdieu Map using specified parameters and a generative model.

        Args:
            llm (t.Optional[str]): The generative model to be used. Default is None.
            x_left_words (t.List[str]): Words defining the left and left x axes.
            x_right_words (t.List[str]): Words defining the left and right x axes.
            y_top_words (t.List[str]): Words defining the left and top y axes.
            y_bottom_words (t.List[str]): Words defining the top and bottom y axes.
            height (int): Dimensions of the visualization. Default to 1500.
            width (int): Dimensions of the visualization. Default to 1500.
            display_percent (bool): Flag to display percentages on the map. Default is True.
            clustering (bool): Whether to apply clustering on the map. Default is False.
            topic_n_clusters (int): Number of clusters for topic modeling. Default is 10.
            topic_terms (int): Length of topic names. Default is 2.
            topic_ngrams (t.List[int]): N-gram range for topic modeling. Default is [1, 2].
            topic_top_terms_overall (int): Top terms to consider overall. Default is 1000.
            gen_topic_language (str): Language for topic generation. Default is "english".
            manual_axis_name (t.Optional[dict]): Custom axis names for the map. Default is None.
            use_doc_gen_topic (bool): Flag to use document context in topic generation. Default is False.
            radius_size (float): Radius size for the map isualization. Default is 0.3.
            convex_hull (bool): Whether to include a convex hull in the visualization. Default is True.
            colorscale (str): colorscale for the Density Plot (Default is delta)
            density (bool): Whether to display a density map

        Returns:
            go.Figure: A Plotly graph object figure representing the Bourdieu Map.

        Note:
            The Bourdieu Map is a sophisticated visualization that plots documents and topics
            based on specified word axes, using a generative model for dynamic analysis.
            This method handles the complex process of generating and plotting this map,
            offering a range of customization options for detailed analysis.
        """

        logger.info("Creating the Bourdieu Map")
        topic_gen_param = TopicGenParam(
            language=gen_topic_language,
            top_doc=3,
            top_terms=10,
            use_doc=use_doc_gen_topic,
            context="everything",
        )

        topic_param = TopicParam(
            n_clusters=topic_n_clusters,
            ngrams=topic_ngrams,
            name_lenght=topic_terms,
            top_terms_overall=topic_top_terms_overall,
        )

        self.bourdieu_query = BourdieuQuery(
            x_left_words=x_left_words,
            x_right_words=x_right_words,
            y_top_words=y_top_words,
            y_bottom_words=y_bottom_words,
            radius_size=radius_size,
        )

        # Request Bourdieu API

        bourdieu_api = BourdieuAPI(
            llm=llm,
            embedding_model=self.embedding_model,
            bourdieu_query=self.bourdieu_query,
            topic_param=topic_param,
            topic_gen_param=topic_gen_param,
            min_docs_per_cluster=min_docs_per_cluster,
        )

        new_docs = copy.deepcopy(self.docs)
        new_terms = copy.deepcopy(self.terms)

        res = bourdieu_api.fit_transform(
            docs=new_docs,
            terms=new_terms,
        )

        self.bourdieu_docs = res[0]
        self.bourdieu_topics = res[1]

        visualizer = BourdieuVisualizer(
            height=height,
            width=width,
            display_percent=display_percent,
            convex_hull=convex_hull,
            clustering=clustering,
            manual_axis_name=manual_axis_name,
            density=density,
            colorscale=colorscale,
            label_size_ratio_clusters=label_size_ratio_clusters,
            label_size_ratio_label=label_size_ratio_label,
            label_size_ratio_percent=label_size_ratio_percent,
        )

        fig = visualizer.fit_transform(self.bourdieu_docs, self.bourdieu_topics)

        return fig

    def visualize_bourdieu_one_dimension(
        self,
        left: t.List[str] = ["negative"],
        right: t.List[str] = ["positive"],
        width: int = 800,
        height: int = 800,
        explainer: bool = False,
    ) -> t.Tuple[go.Figure, t.Union[plt.Figure, None]]:
        """
        Visualizes the document set on a one-dimensional Bourdieu axis.

        Args:
            left (t.List[str]): List of words representing the left side of the axis.
            right (t.List[str]): List of words representing the right side of the axis.
            width (int): Width of the generated visualization. Default is 800.
            height (int): Height of the generated visualization. Default is 800.
            explainer (bool): Flag to include an explainer figure. Default is False.

        Returns:
            t.Tuple[go.Figure, t.Union[plt.Figure, None]]: A tuple containing the main visualization figure
            and an optional explainer figure (if explainer is True).

        Note:
            This method creates a one-dimensional Bourdieu-style visualization, plotting documents along an
            axis defined by contrasting word sets. It helps in understanding the distribution of documents
            in terms of these contrasting word concepts. An optional explainer figure can provide additional
            insight into specific terms used in the visualization.
        """

        model_bourdieu = BourdieuOneDimensionVisualizer(
            embedding_model=self.embedding_model,
            left=left,
            right=right,
            width=width,
            height=height,
            explainer=explainer,
        )

        fig = model_bourdieu.fit_transform(
            docs=self.docs,
        )

        return fig

    def visualize_query(
        self,
        query="What is America?",
        min_score: float = 0.2,
        width: int = 600,
        height: int = 300,
    ):
        # Create a visualization plot using plot_query function
        fig, percent = plot_query(
            embedding_model=self.embedding_model,
            docs=self.docs,
            query=query,
            min_score=min_score,
            width=width,
            height=height,
        )

        # Return the visualization figure and percentage
        return fig, percent

    def visualize_dimensions(
        self,
        dimensions: t.List[str] = ["positive", "negative", "fear", "love"],
        width=500,
        height=500,
        template="plotly_dark",
    ) -> go.Figure:
        """
        Visualizes the similarity scores between a given query and the document set.

        Args:
            width (int): Width of the visualization. Default is 600.
            height (int): Height of the visualization. Default is 300.

        Returns:
            A tuple (fig, percent) where 'fig' is a Plotly graph object figure representing the
            visualization and 'percent' is the percentage of documents above the similarity threshold.

        Note:
            This method creates a visualization showing how closely documents in the set relate to
            the specified query. Documents with similarity scores above the threshold are highlighted,
            providing a visual representation of their relevance to the query.
        """

        final_df = []
        logger.info("Computing Similarities")
        scaler = MinMaxScaler(feature_range=(0, 1))
        for dim in tqdm(dimensions):
            df_search = self.search(dim)
            df_search = self.vectorstore.similarity_search_with_score(dim, k=3)
            df_search["score"] = scaler.fit_transform(
                df_search[["cosine_similarity_score"]]
            )
            df_search["source"] = dim
            final_df.append(df_search)
        final_df = pd.concat([x for x in final_df])

        final_df_mean = (
            final_df.groupby("source")["score"]
            .mean()
            .rename("mean_score")
            .reset_index()
        )
        final_df_mean = final_df_mean.sort_values(
            "mean_score", ascending=True
        ).reset_index(drop=True)
        final_df_mean["rank"] = final_df_mean.index + 1

        self.df_dimensions = final_df_mean

        fig = px.line_polar(
            final_df_mean,
            r="mean_score",
            theta="source",
            line_close=True,
            template=template,
            width=width,
            height=height,
        )
        return fig

    def get_topic_repartition(self, width: int = 1200, height: int = 800) -> go.Figure:
        """
        Creates a bar plot to visualize the distribution of topics by size.

        Args:
            width (int): The width of the bar plot. Default is 1200.
            height (int): The height of the bar plot. Default is 800.

        Returns:
            go.Figure: A Plotly graph object figure representing the topic distribution bar plot.

        Note:
            This method generates a visualization that illustrates the number of documents
            associated with each topic, helping to understand the prevalence and distribution
            of topics within the document set. It provides a clear and concise bar plot for
            easy interpretation of the topic sizes.
        """

        fig = get_topic_repartition(self.topics, width=width, height=height)
        return fig

    def clean_data_by_topics(self):
        """
        Filters and cleans the dataset based on user-selected topics.

        This method presents a UI with checkboxes for each topic in the dataset.
        The user can select topics to keep, and the data will be filtered accordingly.
        It merges the filtered documents and topics data, renames columns for clarity,
        and calculates the percentage of data retained after cleaning.

        Attributes Updated:
            - self.df_cleaned: DataFrame containing the merged and cleaned documents and topics.

        Logging:
            - Logs the percentage of data retained after cleaning.

        Side Effects:
            - Updates `self.df_cleaned` with the cleaned data.
            - Displays interactive widgets for user input.
            - Logs information about the data cleaning process.

        Note:
            - This method uses interactive widgets (checkboxes and a button) for user input.
            - The cleaning process is triggered by clicking the 'Clean Data' button.

        """

        def on_button_clicked(b):
            selected_topics = [
                checkbox.description for checkbox in checkboxes if checkbox.value
            ]
            topic_filtered = [x for x in self.topics if x.name in selected_topics]
            topic_id_filtered = [x.topic_id for x in topic_filtered]
            docs_filtered = [x for x in self.docs if x.topic_id in topic_id_filtered]

            df_docs_cleaned = pd.DataFrame([doc.model_dump() for doc in docs_filtered])
            df_docs_cleaned = df_docs_cleaned[["doc_id", "content", "topic_id"]]
            df_topics = pd.DataFrame([topic.model_dump() for topic in topic_filtered])
            df_topics = df_topics[["topic_id", "name"]]
            self.df_cleaned_ = pd.merge(df_docs_cleaned, df_topics, on="topic_id")
            self.df_cleaned_ = self.df_cleaned_.rename(columns={"name": "topic_name"})

            len_kept = len(docs_filtered)
            len_docs = len(self.docs)
            percent_kept = round(len_kept / len_docs, 2) * 100
            percent_kept = str(percent_kept) + "%"

            logger.info(f"After cleaning, you've kept {percent_kept} of your data")

        # Optionally, return or display df_cleaned
        topic_names = [x.name for x in self.topics]
        checkboxes = [
            Checkbox(description=name, value=True, layout=Layout(width="auto"))
            for name in topic_names
        ]

        title_label = Label("Click on the topics you want to remove 🧹✨🧼🧽")
        checkbox_container = VBox(
            [title_label] + checkboxes, layout=Layout(overflow="scroll hidden")
        )
        button = Button(
            description="Clean Data",
            style={"button_color": "#2596be", "color": "#2596be"},
        )
        button.on_click(on_button_clicked)
        display(checkbox_container, button)

    def manually_clean_topics(self):
        """
        Allows manual renaming of topic names in the dataset.

        This method facilitates the manual editing of topic names based on their IDs.
        If no changes are made, it retains the original topic names.

        The updated topic names are then applied to the `topics` attribute of the class instance.

        Attributes Updated:
            - self.topics: Each topic in this list gets its name updated based on the changes.

        Logging:
            - Logs the percentage of data retained after cleaning.

        Side Effects:
            - Modifies the `name` attribute of each topic in `self.topics` based on user input or defaults.
            - Displays interactive widgets for user input.
            - Logs information about the data cleaning process.

        Note:
            - This method uses interactive widgets (text fields and a button) for user input.
            - The cleaning process is triggered by clicking the 'Apply Changes' button.

        """

        def apply_changes(b):
            for i, text_widget in enumerate(text_widgets):
                new_name = text_widget.value.strip()
                if new_name == "":
                    new_names.append(original_topic_names[i])  # Keep the same name
                else:
                    new_names.append(new_name)

            # Log changes applied
            logger.info("Changes Applied!")

            # Update the topic names
            topic_dict = dict(zip(original_topic_ids, new_names))
            for topic in self.topics:
                topic.name = topic_dict.get(topic.topic_id)

            self.df_topics_, self.df_top_docs_per_topic_ = _create_topic_dfs(
                self.topics, self.docs
            )

        original_topic_names = [x.name for x in self.topics]
        original_topic_ids = [x.topic_id for x in self.topics]
        new_names = []

        # Create a list of Text widgets for entering new names with IDs as descriptions
        text_widgets = []

        for i, (topic, topic_id) in enumerate(
            zip(original_topic_names, original_topic_ids)
        ):
            text_widget = widgets.Text(value=topic, description=f"{topic_id}:")
            text_widgets.append(text_widget)

        # Create a title widget
        title_widget = widgets.HTML("Manually input the new topic names: ")

        # Combine the title, Text widgets, and a button in a VBox
        container = widgets.VBox([title_widget] + text_widgets)

        # Create a button to apply changes with text color #2596be and bold description
        apply_button = widgets.Button(
            description="Apply Changes",
            style={"button_color": "#2596be", "color": "#2596be"},
        )
        apply_button.on_click(apply_changes)

        # Display the container and apply button
        display(container, apply_button)

    def start_server(self):
        subprocess.run(["cp", "web/env.model", "web/.env"], check=True)
        if is_server_running():
            logger.info("Server on port 3000 is already running. Killing it...")
            kill_server()
        if not self.topics:
            raise BunkaError("No topics available. Run bunka.get_topics() first.")
        else:
            file_path = "web/public" + "/bunka_docs.json"

            for x in self.docs:
                x.embedding = None
            docs_json = [x.model_dump() for x in self.docs]
            with open(file_path, "w") as json_file:
                json.dump(docs_json, json_file)

            file_path = "web/public" + "/bunka_topics.json"

            topics_json = [x.model_dump() for x in self.topics]
            with open(file_path, "w") as json_file:
                json.dump(topics_json, json_file)

        """try:
            file_path = "web/public" + "/bunka_bourdieu_docs.json"
            docs_json = [x.model_dump() for x in self.bourdieu_docs]

            with open(file_path, "w") as json_file:
                json.dump(docs_json, json_file)

            file_path = "web/public" + "/bunka_bourdieu_topics.json"
            topics_json = [x.model_dump() for x in self.bourdieu_topics]
            with open(file_path, "w") as json_file:
                json.dump(topics_json, json_file)

            file_path = "web/public" + "/bunka_bourdieu_query.json"
            with open(file_path, "w") as json_file:
                json.dump(self.bourdieu_query.model_dump(), json_file)
        except:
            logger.info("run bunka.visualize_bourdieu() first")"""

        subprocess.Popen(["npm", "start"], cwd="web")
        logger.info("NPM server started.")

    def _quick_plot(self, df_embeddings_2D):
        # Create a scatter plot
        fig_quick_embedding = px.scatter(
            df_embeddings_2D, x="x", y="y", hover_data=["bunka_docs"]
        )

        # Update layout for better readability
        fig_quick_embedding.update_layout(
            title="Raw Scatter Plot of Bunka Embeddings",
            xaxis_title="X Embedding",
            yaxis_title="Y Embedding",
            hovermode="closest",
        )
        # Show the plot

        return fig_quick_embedding
