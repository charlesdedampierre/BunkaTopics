import copy
import copy
import json
import os
import random
import string
import subprocess
import typing as t
import uuid
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from numba.core.errors import NumbaDeprecationWarning
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from bunkatopics.bourdieu import BourdieuAPI, BourdieuOneDimensionVisualizer
from bunkatopics.bourdieu import BourdieuAPI, BourdieuOneDimensionVisualizer
from bunkatopics.datamodel import (
    DOC_ID,
    BourdieuQuery,
    Document,
    Topic,
    TopicGenParam,
    TopicParam,
)
from bunkatopics.logging import logger
from bunkatopics.serveur.server_utils import is_server_running, kill_server
from bunkatopics.topic_modeling import (
    BunkaTopicModeling,
    LLMCleaningTopic,
    TextacyTermsExtractor,
)
from bunkatopics.topic_modeling.coherence_calculator import get_coherence
from bunkatopics.topic_modeling.document_topic_analyzer import get_top_documents
from bunkatopics.topic_modeling.topic_utils import get_topic_repartition
from bunkatopics.visualization import BourdieuVisualizer, TopicVisualizer
from bunkatopics.visualization import BourdieuVisualizer, TopicVisualizer
from bunkatopics.visualization.query_visualizer import plot_query

import warnings

import warnings

# Filter ResourceWarning
warnings.filterwarnings("ignore")
# Filter ResourceWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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

    def __init__(self, embedding_model=None, language: str = "english"):
        """Initialize a BunkaTopics instance.

        Arguments:
           embedding_model : An optional embedding model for generating document embeddings.
               If not provided, a default model will be used based on the specified language.
               Default: None
           language : The language to be used for text processing and modeling.
               Options: "english" (default), or specify another language as needed.
               Default: "english"
        """
        if embedding_model is None:
            if language == "english":
                embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            else:
                embedding_model = HuggingFaceEmbeddings(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
        self.embedding_model = embedding_model
        self.language = language

    def fit(
        self,
        docs: t.List[str],
        ids: t.List[DOC_ID] = None,
    ) -> None:
        """
        Fits the Bunka model to the provided list of documents.

        This method processes the documents, extracts terms, generates embeddings, and
        applies dimensionality reduction to prepare the data for topic modeling.

        Args:
            docs (t.List[str]): A list of document strings.
            ids (t.Optional[t.List[DOC_ID]]): Optional. A list of identifiers for the documents. If not provided, UUIDs are generated.
        """

        df = pd.DataFrame(docs, columns=["content"])

        # Transform into a Document model
        if ids is not None:
            df["doc_id"] = ids
        else:
            df["doc_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
        df = df[~df["content"].isna()]
        df = df.reset_index(drop=True)
        self.docs = [Document(**row) for row in df.to_dict(orient="records")]
        sentences = [doc.content for doc in self.docs]
        ids = [doc.doc_id for doc in self.docs]

        logger.info("Extracting meaningful terms from documents...")
        terms_extractor = TextacyTermsExtractor(language=self.language)
        self.terms, indexed_terms_dict = terms_extractor.fit_transform(ids, sentences)

        # add to the docs object
        for doc in self.docs:
            doc.term_id = indexed_terms_dict.get(doc.doc_id, [])

        logger.info(
            "Embedding documents... (can take varying amounts of time depending on their size)"
        )

        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choice(characters) for _ in range(20))

        df_loader = pd.DataFrame(sentences, columns=["text"])
        df_loader["doc_id"] = ids

        loader = DataFrameLoader(df_loader, page_content_column="text")
        documents_langchain = loader.load()
        self.vectorstore = Chroma.from_documents(
            documents_langchain, self.embedding_model, collection_name=random_string
        )

        bunka_ids = [item["doc_id"] for item in self.vectorstore.get()["metadatas"]]
        bunka_docs = self.vectorstore.get()["documents"]
        bunka_embeddings = self.vectorstore._collection.get(include=["embeddings"])[
            "embeddings"
        ]

        # Add to the bunka objects
        emb_doc_dict = {x: y for x, y in zip(bunka_ids, bunka_embeddings)}
        for doc in self.docs:
            doc.embedding = emb_doc_dict.get(doc.doc_id, [])

        logger.info("Reducing the dimensions of embeddings...")
        reducer = umap.UMAP(
            n_components=2,
            random_state=None,
        )  # Not random state to go quicker
        bunka_embeddings_2D = reducer.fit_transform(bunka_embeddings)
        df_embeddings_2D = pd.DataFrame(bunka_embeddings_2D, columns=["x", "y"])
        df_embeddings_2D["doc_id"] = bunka_ids
        df_embeddings_2D["bunka_docs"] = bunka_docs

        xy_dict = df_embeddings_2D.set_index("doc_id")[["x", "y"]].to_dict("index")

        # Update the documents with the x and y values from the DataFrame
        for doc in self.docs:
            doc.x = xy_dict[doc.doc_id]["x"]
            doc.y = xy_dict[doc.doc_id]["y"]

        self.df_embeddings_2D = df_embeddings_2D

        # Create a scatter plot
        fig_quick_embedding = px.scatter(
            self.df_embeddings_2D, x="x", y="y", hover_data=["bunka_docs"]
        )

        # Update layout for better readability
        fig_quick_embedding.update_layout(
            title="Raw Scatter Plot of Bunka Embeddings",
            xaxis_title="X Embedding",
            yaxis_title="Y Embedding",
            hovermode="closest",
        )
        # Show the plot
        self.fig_quick_embedding = fig_quick_embedding

    def fit_transform(self, docs: t.List[Document], n_clusters=3) -> pd.DataFrame:
        self.fit(docs)
        df_topics = self.get_topics(n_clusters=n_clusters)
        return df_topics

    def get_topics(
        self,
        n_clusters: int = 5,
        ngrams: t.List[int] = [1, 2],
        name_length: int = 10,
        top_terms_overall: int = 2000,
        min_count_terms: int = 2,
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

        Returns:
            pd.DataFrame: A DataFrame containing the topics and their associated data.

        Note:
            The method applies topic modeling using the specified parameters and updates the internal state
            with the resulting topics. It also associates the identified topics with the documents.
        """

        logger.info("Computing the topics")

        topic_model = BunkaTopicModeling(
            n_clusters=n_clusters,
            ngrams=ngrams,
            name_length=name_length,
            x_column="x",
            y_column="y",
            top_terms_overall=top_terms_overall,
            min_count_terms=min_count_terms,
        )

        self.topics: t.List[Topic] = topic_model.fit_transform(
            docs=self.docs,
            terms=self.terms,
        )

        self.docs, self.topics = get_top_documents(
            self.docs, self.topics, ranking_terms=20
        )
        df_topics = pd.DataFrame.from_records(
            [topic.model_dump() for topic in self.topics]
        )
        df_topics = pd.DataFrame.from_records(
            [topic.model_dump() for topic in self.topics]
        )
        return df_topics

    def get_clean_topic_name(
        self,
        llm,
        language: str = "english",
        use_doc: bool = False,
        context: str = "everything",
    ) -> pd.DataFrame:
        """
        Enhances topic names using a language model for cleaner and more meaningful representations.

        Args:
            llm: The language model used for cleaning topic names.
            language (str): The language context for the language model. Default is "english".
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
            language=language,
            use_doc=use_doc,
            context=context,
        )
        self.topics: t.List[Topic] = model_cleaning.fit_transform(
            self.topics,
            self.docs,
        )

        df_topics = pd.DataFrame.from_records(
            [topic.model_dump() for topic in self.topics]
        )
        df_topics = pd.DataFrame.from_records(
            [topic.model_dump() for topic in self.topics]
        )

        return df_topics

    def visualize_topics(
        self,
        show_text: bool = True,
        label_size_ratio: int = 100,
        width: int = 1000,
        height: int = 1000,
    ) -> go.Figure:
        """
        Generates a visualization of the identified topics in the document set.

        Args:
            show_text (bool): Whether to display text labels on the visualization. Default is True.
            label_size_ratio (int): The size ratio of the labels in the visualization. Default is 100.
            width (int): The width of the visualization figure. Default is 1000.
            height (int): The height of the visualization figure. Default is 1000.

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
        )
        fig = model_visualizer.fit_transform(
            self.docs,
            self.topics,
        )

        fig = model_visualizer.fit_transform(
            self.docs,
            self.topics,
        )

        return fig

    def visualize_bourdieu(
        self,
        llm: t.Optional[str] = None,
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
        topic_gen_name: bool = False,
        manual_axis_name: t.Optional[dict] = None,
        use_doc_gen_topic: bool = False,
        radius_size: float = 0.3,
        convex_hull: bool = True,
    ) -> go.Figure:
        """
        Creates and visualizes a Bourdieu Map using specified parameters and a generative model.

        Args:
            generative_model (t.Optional[str]): The generative model to be used. Default is None.
            x_left_words, x_right_words (t.List[str]): Words defining the left and right axes.
            y_top_words, y_bottom_words (t.List[str]): Words defining the top and bottom axes.
            height, width (int): Dimensions of the visualization. Both default to 1500.
            display_percent (bool): Flag to display percentages on the map. Default is True.
            clustering (bool): Whether to apply clustering on the map. Default is False.
            topic_n_clusters (int): Number of clusters for topic modeling. Default is 10.
            topic_terms (int): Length of topic names. Default is 2.
            topic_ngrams (t.List[int]): N-gram range for topic modeling. Default is [1, 2].
            topic_top_terms_overall (int): Top terms to consider overall. Default is 1000.
            gen_topic_language (str): Language for topic generation. Default is "english".
            topic_gen_name (bool): Flag to generate topic names. Default is False.
            manual_axis_name (t.Optional[dict]): Custom axis names for the map. Default is None.
            use_doc_gen_topic (bool): Flag to use document context in topic generation. Default is False.
            radius_size (float): Radius size for the map isualization. Default is 0.3.
            convex_hull (bool): Whether to include a convex hull in the visualization. Default is True.
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
            generative_ai_name=topic_gen_name,
            topic_param=topic_param,
            topic_gen_param=topic_gen_param,
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
        )

        fig = visualizer.fit_transform(self.bourdieu_docs, self.bourdieu_topics)

        return fig

    def rag_query(self, query: str, llm, top_doc: int = 2):
        """
        Executes a Retrieve-and-Generate (RAG) query using the provided language model and document set.

        Args:
            query (str): The query string to be processed.
            llm: The language model used for generating answers.
            top_doc (int): The number of top documents to retrieve for the query. Default is 2.

        Returns:
            The response from the RAG query, including the answer and source documents.

        Note:
            This method utilizes a RetrievalQA chain to answer queries. It retrieves relevant documents
            based on the query and uses the language model to generate a response. The method is designed
            to work with complex queries and provide informative answers using the document set.
        """
        # Log a message indicating the query is being processed
        logger.info("Answering your query, please wait a few seconds")

        # Create a RetrievalQA instance with the specified llm and retriever
        qa_with_sources_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_doc}),
            return_source_documents=True,  # Include source documents in the response
        )

        # Provide the query to the RetrievalQA instance for answering
        response = qa_with_sources_chain({"query": query})

        return response

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

        fig, fig_specific_terms = model_bourdieu.fit_transform(
            docs=self.docs,
        )

        fig, fig_specific_terms = model_bourdieu.fit_transform(
            docs=self.docs,
        )

        return fig, fig_specific_terms

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
            query (str): The query to be visualized against the documents. Default is "What is America?".
            min_score (float): The minimum similarity score threshold for visualization. Default is 0.2.
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

    def get_topic_coherence(self, topic_terms_n=10):
        texts = [doc.term_id for doc in self.docs]
        res = get_coherence(self.topics, texts, topic_terms_n=topic_terms_n)
        return res

    def start_server_bourdieu(self):
        if is_server_running():
            print("Server on port 3000 is already running. Killing it...")
            kill_server()
        try:
            file_path = "../web/public" + "/bunka_bourdieu_docs.json"
            docs_json = [x.model_dump() for x in self.bourdieu_docs]
            docs_json = [x.model_dump() for x in self.bourdieu_docs]

            with open(file_path, "w") as json_file:
                json.dump(docs_json, json_file)

            file_path = "../web/public" + "/bunka_bourdieu_topics.json"
            topics_json = [x.model_dump() for x in self.bourdieu_topics]
            topics_json = [x.model_dump() for x in self.bourdieu_topics]
            with open(file_path, "w") as json_file:
                json.dump(topics_json, json_file)

            file_path = "../web/public" + "/bunka_bourdieu_query.json"
            with open(file_path, "w") as json_file:
                json.dump(self.bourdieu_query.model_dump(), json_file)
                json.dump(self.bourdieu_query.model_dump(), json_file)

            subprocess.Popen(["npm", "start"], cwd="../web")
            print(
                "NPM server started. Please Switch to Bourdieu View to see the results"
            )
        except Exception as e:
            print(f"Error starting NPM server: {e}")

    def start_server(self):
        if is_server_running():
            print("Server on port 3000 is already running. Killing it...")
            kill_server()
        try:
            file_path = "../web/public" + "/bunka_docs.json"
            docs_json = [x.model_dump() for x in self.docs]
            docs_json = [x.model_dump() for x in self.docs]

            with open(file_path, "w") as json_file:
                json.dump(docs_json, json_file)

            file_path = "../web/public" + "/bunka_topics.json"
            topics_json = [x.model_dump() for x in self.topics]
            topics_json = [x.model_dump() for x in self.topics]
            with open(file_path, "w") as json_file:
                json.dump(topics_json, json_file)

            subprocess.Popen(["npm", "start"], cwd="../web")
            print("NPM server started.")
        except Exception as e:
            print(f"Error starting NPM server: {e}")
