import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
import pandas as pd
from .bunka_logger import logger
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import umap
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import warnings
import plotly.graph_objects as go
from .datamodel import Document, Term, Topic, DOC_ID, TOPIC_ID, TERM_ID
from .functions.topic_document import get_top_documents
from .functions.topic_utils import get_topic_repartition
from .visualisation.bourdieu import visualize_bourdieu
from .visualisation.topic_visualization import visualize_topics
from .functions.extract_terms import extract_terms_df
from .functions.topic_gen_representation import get_df_prompt, get_clean_topics
from .functions.topics_modeling import get_topics
from .visualisation.bourdieu import visualize_bourdieu_one_dimension

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
    def __init__(self, embedding_model=None, language: str = "en_core_web_sm"):
        if embedding_model is None:
            embedding_model = HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-large",
                embed_instruction="Embed the documents for visualisation of Topic Modeling on a map : ",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        self.embedding_model = embedding_model
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
        else:
            df["doc_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
        df = df[~df["content"].isna()]
        df = df.reset_index(drop=True)

        self.docs = [Document(**row) for row in df.to_dict(orient="records")]
        df = pd.DataFrame.from_records([doc.dict() for doc in self.docs])

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
        for doc in self.docs:
            doc.term_id = indexed_terms_dict.get(doc.doc_id, [])

        # Embed sentences

        logger.info("Embedding Documents, this may take few minutes")

        sentences = [doc.content for doc in self.docs]
        ids = [doc.doc_id for doc in self.docs]

        # Using FAISS to index and embed the documents

        df_temporary = pd.DataFrame(sentences)
        loader = DataFrameLoader(df_temporary, page_content_column=0)
        documents_langchain = loader.load()

        # load it into Chroma
        vectorstore = Chroma.from_documents(documents_langchain, self.embedding_model)
        self.vectorstore = vectorstore

        # Get all embeddings
        embeddings = vectorstore._collection.get(include=["embeddings"])["embeddings"]
        final_ids = vectorstore._collection.get(include=["embeddings"])["ids"]

        """
        vectorstore = FAISS.from_texts(
            texts=sentences, ids=ids, embedding=self.embedding_model
        )

        self.vectorstore = vectorstore
        embeddings = vectorstore.index.reconstruct_n(0, len(sentences))

        """

        # embeddings = self.embedding_model.embed_documents(sentences)

        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.index = final_ids

        emb_doc_dict = {x: y for x, y in zip(ids, embeddings)}

        for doc in self.docs:
            doc.embedding = emb_doc_dict.get(doc.doc_id, [])

        logger.info("Reducing Dimensions")
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2D = reducer.fit_transform(embeddings)

        df_embeddings_2D = pd.DataFrame(embeddings_2D)
        df_embeddings_2D.columns = ["x", "y"]
        df_embeddings_2D["doc_id"] = ids

        xy_dict = df_embeddings_2D.set_index("doc_id")[["x", "y"]].to_dict("index")

        # Update the documents with the x and y values from the DataFrame
        for doc in self.docs:
            doc.x = xy_dict[doc.doc_id]["x"]
            doc.y = xy_dict[doc.doc_id]["y"]

        self.terms: t.List[Term] = terms

    def fit_transform(self, docs: t.List[Document], n_clusters=40) -> pd.DataFrame:
        self.fit(docs)
        df_topics = self.get_topics(n_clusters=n_clusters)
        return df_topics

    def get_topics(self, n_clusters=5, ngrams=[1, 2], name_lenght=15) -> pd.DataFrame:
        self.topics: t.List[Topic] = get_topics(
            docs=self.docs,
            terms=self.terms,
            n_clusters=n_clusters,
            ngrams=ngrams,
            name_lenght=name_lenght,
            x_column="x",
            y_column="y",
        )

        self.topics: t.List[Topic] = get_top_documents(
            self.docs, self.topics, ranking_terms=20, top_docs=5
        )
        df_topics = pd.DataFrame.from_records([topic.dict() for topic in self.topics])
        return df_topics

    def rag_query(self, query: str, generative_model, top_doc: int = 2):
        logger.info("Answering your query, please wait a few seconds")

        # this is the entire retrieval system
        qa_with_sources_chain = RetrievalQA.from_chain_type(
            llm=generative_model,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": top_doc}),
            # chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )

        response = qa_with_sources_chain({"query": query})

        return response

    def get_clean_topic_name(self, generative_model) -> pd.DataFrame:
        """

        Get the topic name using Generative AI

        """

        from bunkatopics.functions.topic_gen_representation import get_clean_topic_all

        self.topics: t.List[Topic] = get_clean_topic_all(
            generative_model, self.topics, self.docs
        )
        df_topics = pd.DataFrame.from_records([topic.dict() for topic in self.topics])

        return df_topics

    def search(self, user_input: str, top_doc: int = 3) -> pd.DataFrame:
        res = self.vectorstore.similarity_search_with_score(user_input, k=top_doc)
        # res = vector_search(self.docs, self.embedding_model, user_input=user_input)
        return res

    def get_topic_coherence(self, topic_terms_n=10):
        texts = [doc.term_id for doc in self.docs]
        res = get_coherence(self.topics, texts, topic_terms_n=topic_terms_n)
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
        openai_key=None,
        height=1500,
        width=1500,
        label_size_ratio_label=50,
        display_percent=True,
        clustering=False,
        topic_n_clusters=10,
        topic_terms=2,
        topic_top_terms_overall=500,
        topic_gen_name=False,
    ) -> go.Figure:
        fig, self.df_bourdieu = visualize_bourdieu(
            self.embedding_model,
            docs=self.docs,
            terms=self.terms,
            openai_key=openai_key,
            x_left_words=x_left_words,
            x_right_words=x_right_words,
            y_top_words=y_top_words,
            y_bottom_words=y_bottom_words,
            height=height,
            width=width,
            display_percent=display_percent,
            label_size_ratio_label=label_size_ratio_label,
            clustering=clustering,
            topic_gen_name=topic_gen_name,
            topic_n_clusters=topic_n_clusters,
            topic_terms=topic_terms,
            topic_top_terms_overall=topic_top_terms_overall,
        )

        return fig

    def visualize_topics(
        self, add_scatter=False, label_size_ratio=100, width=1000, height=1000
    ) -> go.Figure:
        fig = visualize_topics(
            self.docs,
            self.topics,
            width=width,
            height=height,
            add_scatter=add_scatter,
            label_size_ratio=label_size_ratio,
        )
        return fig

    def visualize_bourdieu_one_dimension(
        self,
        left=["negative", "bad"],
        right=["positive"],
        width=1200,
        height=1200,
        explainer=True,
    ):
        fig = visualize_bourdieu_one_dimension(
            docs=self.docs,
            embedding_model=self.embedding_model,
            left=left,
            right=right,
            width=width,
            height=height,
            explainer=explainer,
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
