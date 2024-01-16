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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from numba.core.errors import NumbaDeprecationWarning
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from bunkatopics.datamodel import (DOC_ID, BourdieuQuery, Document, Topic,
                                   TopicGenParam, TopicParam)
from bunkatopics.logging import logger
from bunkatopics.serveur.server_utils import is_server_running, kill_server
from bunkatopics.topic_modeling import BourdieuAPI, BunkaTopicModeling
from bunkatopics.topic_modeling.coherence_calculator import get_coherence
from bunkatopics.topic_modeling.document_topic_analyzer import \
    get_top_documents
from bunkatopics.topic_modeling.llm_topic_representation import \
    get_clean_topic_all
from bunkatopics.topic_modeling.term_extractor import TextacyTermsExtractor
from bunkatopics.topic_modeling.topic_utils import get_topic_repartition
from bunkatopics.visualization.bourdieu_visualizer import (
    visualize_bourdieu, visualize_bourdieu_one_dimension)
from bunkatopics.visualization.query_visualizer import plot_query
from bunkatopics.visualization.topic_visualizer import visualize_topics

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

        logger.info("Extracting terms from documents")
        terms_extractor = TextacyTermsExtractor(language=self.language)
        self.terms, indexed_terms_dict = terms_extractor.fit_transform(ids, sentences)

        # add to the docs object
        for doc in self.docs:
            doc.term_id = indexed_terms_dict.get(doc.doc_id, [])

        logger.info("Embedding Documents, this may time depending on the size")

        characters = string.ascii_letters + string.digits
        random_string = "".join(random.choice(characters) for _ in range(20))

        # using Chroma as a vectorstore
        self.vectorstore = Chroma(
            embedding_function=self.embedding_model, collection_name=random_string
        )

        self.vectorstore.add_texts(texts=sentences, ids=ids)
        embeddings = self.vectorstore._collection.get(include=["embeddings"])[
            "embeddings"
        ]

        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.index = ids

        emb_doc_dict = {x: y for x, y in zip(ids, embeddings)}

        for doc in self.docs:
            doc.embedding = emb_doc_dict.get(doc.doc_id, [])

        logger.info("Reducing Dimensions")

        reducer = umap.UMAP(
            n_components=2,
            random_state=None,
        )  # Not random state to go quicker
        embeddings_2D = reducer.fit_transform(embeddings)
        df_embeddings_2D = pd.DataFrame(embeddings_2D)
        df_embeddings_2D.columns = ["x", "y"]
        df_embeddings_2D["doc_id"] = ids

        xy_dict = df_embeddings_2D.set_index("doc_id")[["x", "y"]].to_dict("index")

        # Update the documents with the x and y values from the DataFrame
        for doc in self.docs:
            doc.x = xy_dict[doc.doc_id]["x"]
            doc.y = xy_dict[doc.doc_id]["y"]

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
        df_topics = pd.DataFrame.from_records([topic.dict() for topic in self.topics])
        return df_topics

    def get_clean_topic_name(
        self,
        llm,
        language: str = "english",
        use_doc: bool = False,
        context: str = "everything",
    ) -> pd.DataFrame:
        logger.info("Using LLM to make topic names cleaner")
        self.topics: t.List[Topic] = get_clean_topic_all(
            llm,
            self.topics,
            self.docs,
            language=language,
            use_doc=use_doc,
            context=context,
        )
        df_topics = pd.DataFrame.from_records([topic.dict() for topic in self.topics])

        return df_topics

    def visualize_topics(
        self,
        show_text: bool = True,
        label_size_ratio: int = 100,
        width: int = 1000,
        height: int = 1000,
    ) -> go.Figure:
        logger.info("Creating the Bunka Map")
        fig = visualize_topics(
            self.docs,
            self.topics,
            width=width,
            height=height,
            show_text=show_text,
            label_size_ratio=label_size_ratio,
        )
        return fig

    def visualize_bourdieu(
        self,
        generative_model: t.Optional[str] = None,
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
            generative_model=generative_model,
            embedding_model=self.embedding_model,
            bourdieu_query=self.bourdieu_query,
            generative_ai_name=topic_gen_name,
            topic_param=topic_param,
            topic_gen_param=topic_gen_param,
        )

        res = bourdieu_api.fit_transform(
            docs=self.docs,
            terms=self.terms,
        )

        self.bourdieu_docs = res[0]
        self.bourdieu_topics = res[1]

        # Visualize The results from the API
        fig = visualize_bourdieu(
            self.bourdieu_docs,
            self.bourdieu_topics,
            height=height,
            width=width,
            display_percent=display_percent,
            convex_hull=convex_hull,
            clustering=clustering,
            manual_axis_name=manual_axis_name,
        )

        return fig

    def rag_query(self, query: str, llm, top_doc: int = 2):
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
        fig, fig_specific_terms = visualize_bourdieu_one_dimension(
            docs=self.docs,
            embedding_model=self.embedding_model,
            left=left,
            right=right,
            width=width,
            height=height,
            explainer=explainer,
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
            docs_json = [x.dict() for x in self.bourdieu_docs]

            with open(file_path, "w") as json_file:
                json.dump(docs_json, json_file)

            file_path = "../web/public" + "/bunka_bourdieu_topics.json"
            topics_json = [x.dict() for x in self.bourdieu_topics]
            with open(file_path, "w") as json_file:
                json.dump(topics_json, json_file)

            file_path = "../web/public" + "/bunka_bourdieu_query.json"
            with open(file_path, "w") as json_file:
                json.dump(self.bourdieu_query.dict(), json_file)

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
            docs_json = [x.dict() for x in self.docs]

            with open(file_path, "w") as json_file:
                json.dump(docs_json, json_file)

            file_path = "../web/public" + "/bunka_topics.json"
            topics_json = [x.dict() for x in self.topics]
            with open(file_path, "w") as json_file:
                json.dump(topics_json, json_file)

            subprocess.Popen(["npm", "start"], cwd="../web")
            print("NPM server started.")
        except Exception as e:
            print(f"Error starting NPM server: {e}")
