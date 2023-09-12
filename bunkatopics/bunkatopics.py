import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
import pandas as pd
from .bunka_logger import logger
from langchain.embeddings import HuggingFaceInstructEmbeddings
import umap
from sklearn.cluster import KMeans
import warnings
import plotly.graph_objects as go
from .datamodel import Document, Term, Topic, DOC_ID, TOPIC_ID, TERM_ID
from .functions.topic_document import get_top_documents
from .functions.topic_utils import get_topic_repartition
from .visualisation.bourdieu import visualize_bourdieu
from .visualisation.topic_visualization import visualize_topics
from .functions.extract_terms import extract_terms_df
from .functions.topic_representation import remove_overlapping_terms
from .functions.utils import specificity
from .functions.topic_gen_representation import get_df_prompt, get_clean_topics

from .functions.coherence import get_coherence
from .functions.search import vector_search
import uuid
import typing as t
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Bunka:
    def __init__(self, model_hf=None, language: str = "en_core_web_sm"):
        if model_hf is None:
            model_hf = HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-large",
                embed_instruction="Embed the documents for visualisation of Topic Modeling on a map : ",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        self.model_hf = model_hf
        self.language = language

    def fit(
        self,
        docs: t.List[str],
        ids: t.List[DOC_ID] = None,
        multiprocess: bool = True,
    ) -> None:
        df = pd.DataFrame(docs, columns=["content"])

        if ids is not None:
            df["doc_id"] = ids

        df["doc_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
        df = df[~df["content"].isna()]
        df = df.reset_index(drop=True)

        docs = [Document(**row) for row in df.to_dict(orient="records")]
        df = pd.DataFrame.from_records([doc.dict() for doc in docs])

        logger.info("Extracting Terms")
        df_terms, df_terms_indexed = extract_terms_df(
            df,
            text_var="content",
            index_var="doc_id",
            ngs=True,
            ents=True,
            ncs=True,
            multiprocess=multiprocess,
            sample_size=100000,
            drop_emoji=True,
            ngrams=(1, 2, 3),
            remove_punctuation=True,
            include_pos=["NOUN"],
            include_types=["PERSON", "ORG"],
            language=self.language,
        )

        df_terms = df_terms.reset_index()
        df_terms = df_terms.rename(columns={"terms_indexed": "term_id"})

        terms = [Term(**row) for row in df_terms.to_dict(orient="records")]

        df_terms_indexed = df_terms_indexed.reset_index()
        df_terms_indexed.columns = ["doc_id", "indexed_terms"]
        indexed_terms_dict = df_terms_indexed.set_index("doc_id")[
            "indexed_terms"
        ].to_dict()

        # add to the docs object
        for doc in docs:
            doc.term_id = indexed_terms_dict.get(doc.doc_id, [])

        sentences = [doc.content for doc in docs]
        ids = [doc.doc_id for doc in docs]

        logger.info("Embedding Documents, this may take few minutes")
        embeddings = self.model_hf.embed_documents(sentences)

        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.index = ids

        emb_doc_dict = {x: y for x, y in zip(ids, embeddings)}

        for doc in docs:
            doc.embedding = emb_doc_dict.get(doc.doc_id, [])

        logger.info("Reducing Dimensions")
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2D = reducer.fit_transform(embeddings)

        df_embeddings_2D = pd.DataFrame(embeddings_2D)
        df_embeddings_2D.columns = ["x", "y"]
        df_embeddings_2D["doc_id"] = ids

        xy_dict = df_embeddings_2D.set_index("doc_id")[["x", "y"]].to_dict("index")

        # Update the documents with the x and y values from the DataFrame
        for doc in docs:
            doc.x = xy_dict[doc.doc_id]["x"]
            doc.y = xy_dict[doc.doc_id]["y"]

        self.docs = docs
        self.terms = terms

    def get_topics(self, n_clusters=40, ngrams=[1, 2], name_lenght=15):
        clustering_model = KMeans(n_clusters=n_clusters)
        df_embeddings_2D = pd.DataFrame(
            {
                "doc_id": [doc.doc_id for doc in self.docs],
                "x": [doc.x for doc in self.docs],
                "y": [doc.y for doc in self.docs],
            }
        )

        df_embeddings_2D = df_embeddings_2D.set_index("doc_id")

        df_embeddings_2D["topic_number"] = clustering_model.fit(
            df_embeddings_2D
        ).labels_.astype(str)

        df_embeddings_2D["topic_id"] = "bt" + "-" + df_embeddings_2D["topic_number"]

        # insert into the documents
        topic_doc_dict = df_embeddings_2D["topic_id"].to_dict()
        for doc in self.docs:
            doc.topic_id = topic_doc_dict.get(doc.doc_id, [])

        df_terms = pd.DataFrame.from_records([term.dict() for term in self.terms])
        df_terms = df_terms.sort_values("count_terms", ascending=False)
        df_terms = df_terms.head(1000)
        df_terms = df_terms[df_terms["ngrams"].isin(ngrams)]

        df_terms_indexed = pd.DataFrame.from_records([doc.dict() for doc in self.docs])
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

        df_topics_docs = pd.DataFrame.from_records([doc.dict() for doc in self.docs])
        df_topics_docs = df_topics_docs[["doc_id", "x", "y", "topic_id"]]
        df_topics_docs = df_topics_docs.groupby("topic_id").agg(
            size=("doc_id", "count"), x_centroid=("x", "mean"), y_centroid=("y", "mean")
        )

        topic_dict = df_topics_docs[["size", "x_centroid", "y_centroid"]].to_dict(
            "index"
        )

        # Update the documents with the x and y values from the DataFrame
        for topic in topics:
            topic.size = topic_dict[topic.topic_id]["size"]
            topic.x_centroid = topic_dict[topic.topic_id]["x_centroid"]
            topic.y_centroid = topic_dict[topic.topic_id]["y_centroid"]

        self.topics = topics
        df_topics = pd.DataFrame.from_records([topic.dict() for topic in topics])

        return df_topics

    def get_clean_topic_name(self, openai_key: str):
        """

        Get the topic name using Generative AI

        """
        df_prompt = get_df_prompt(self)
        df_clean = get_clean_topics(df_prompt, openai_key=openai_key)

        topics = list(df_clean["topic_id"])
        names = list(df_clean["topic_gen_name"])
        dict_topic_gen_name = {x: y for x, y in zip(topics, names)}

        for topic in self.topics:
            topic.name = dict_topic_gen_name.get(topic.topic_id, [])

        return df_clean

    def fit_transform(self, docs, n_clusters=40):
        self.fit(docs)
        df_topics = self.get_topics(n_clusters=n_clusters)
        return df_topics

    def search(self, user_input: str):
        res = vector_search(self.docs, self.model_hf, user_input=user_input)
        return res

    def get_topic_coherence(self, topic_terms_n=10):
        texts = [doc.term_id for doc in self.docs]
        res = get_coherence(self.topics, texts, topic_terms_n=topic_terms_n)
        return res

    def get_top_documents(self, top_docs=5, ranking_terms=20) -> pd.DataFrame:
        res = get_top_documents(
            self.docs, self.topics, ranking_terms=ranking_terms, top_docs=top_docs
        )
        df_top_doc = res.groupby("topic_id")["doc_id"].apply(lambda x: list(x))
        top_doc_topic_dict = df_top_doc.to_dict()

        for topic in self.topics:
            topic.top_doc_id = top_doc_topic_dict.get(topic.topic_id, [])

        return res

    def get_topic_repartition(self, width=1200, height=800) -> go.Figure:
        fig = get_topic_repartition(self.topics, width=width, height=height)
        return fig

    def visualize_bourdieu(
        self,
        x_left_words=["war"],
        x_right_words=["peace"],
        y_top_words=["men"],
        y_bottom_words=["women"],
        height=1500,
        width=1500,
        clustering=False,
        n_clusters=10,
        display_percent=True,
    ) -> go.Figure:
        fig, self.df_bourdieu = visualize_bourdieu(
            self.model_hf,
            self.docs,
            x_left_words=x_left_words,
            x_right_words=x_right_words,
            y_top_words=y_top_words,
            y_bottom_words=y_bottom_words,
            height=height,
            width=width,
            clustering=clustering,
            n_clusters=n_clusters,
            display_percent=display_percent,
        )

        return fig

    def visualize_topics(self, add_scatter=False, width=1000, height=1000) -> go.Figure:
        fig = visualize_topics(
            self.docs, self.topics, width=width, height=height, add_scatter=add_scatter
        )
        return fig

    def get_dimensions(
        self, dimensions: t.List[str], width=500, height=500, template="plotly_dark"
    ) -> go.Figure:
        final_df = []
        logger.info("Computing Similarities")
        scaler = MinMaxScaler(feature_range=(0, 1))
        for dim in tqdm(dimensions):
            df_search = self.search(dim)
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
