import typing as t

import numpy as np
import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from bunkatopics.datamodel import (BourdieuDimension, BourdieuQuery,
                                   ContinuumDimension, Document, Term, Topic,
                                   TopicGenParam, TopicParam)
from bunkatopics.topic_modeling import (BunkaTopicModeling, DocumentRanker,
                                        LLMCleaningTopic)

pd.options.mode.chained_assignment = None


class BourdieuAPI:
    """
    A class for performing Bourdieu analysis on a collection of documents.

    This class leverages an embedding model to compute Bourdieu dimensions and topics
    for the given documents. It supports customization of the analysis through various parameters
    and the use of generative AI for topic naming.

    """

    def __init__(
        self,
        embedding_model: Embeddings,
        llm: t.Optional[LLM] = None,
        bourdieu_query: BourdieuQuery = BourdieuQuery(),
        topic_param: TopicParam = TopicParam(),
        topic_gen_param: TopicGenParam = TopicGenParam(),
        min_count_terms: int = 2,
        ranking_terms: int = 20,
    ) -> None:
        """
        Initializes the BourdieuAPI with the provided models, parameters, and configurations.

        Arguments
            llm: The generative AI model for topic naming.
            embedding_model: The model used for embedding documents.
            bourdieu_query (BourdieuQuery, optional): Configuration for Bourdieu analysis.
                                                       Defaults to BourdieuQuery().
            topic_param (TopicParam, optional): Parameters for topic modeling. Defaults to TopicParam().
            generative_ai_name (bool, optional): Flag to use AI for generating topic names. Defaults to False.
            topic_gen_param (TopicGenParam, optional): Parameters for the generative AI in topic naming.
                                                       Defaults to TopicGenParam().
            min_count_terms (int, optional): Minimum term count for topic modeling. Defaults to 2.
        """

        self.llm = llm
        self.embedding_model = embedding_model
        self.bourdieu_query = bourdieu_query
        self.topic_param = topic_param
        self.topic_gen_param = topic_gen_param
        self.min_count_terms = min_count_terms
        self.ranking_terms = ranking_terms

    def fit_transform(
        self, docs: t.List[Document], terms: t.List[Term]
    ) -> t.Tuple[t.List[Document], t.List[Topic]]:
        """
        Processes the documents and terms to compute Bourdieu dimensions and topics.

        This method applies the embedding model to compute Bourdieu dimensions for each document
        based on provided queries. It also performs topic modeling on the documents and, if enabled,
        uses a generative AI model for naming the topics.

        Arguments:
            docs (List[Document]): List of Document objects representing the documents to be analyzed.
            terms (List[Term]): List of Term objects representing the terms to be used in topic modeling.

        Notes:
            - The method first resets Bourdieu dimensions for all documents.
            - It computes Bourdieu continuums based on the configured left and right words.
            - Documents are then filtered based on their position relative to a defined radius in the Bourdieu space.
            - Topic modeling is performed on the filtered set of documents.
            - If `generative_ai_name` is True, topics are named using the generative AI model.
        """

        # Reset Bourdieu dimensions for all documents
        for doc in docs:
            doc.bourdieu_dimensions = []

        # Compute Continuums
        new_docs = _get_continuum(
            self.embedding_model,
            docs,
            cont_name="cont1",
            left_words=self.bourdieu_query.x_left_words,
            right_words=self.bourdieu_query.x_right_words,
        )
        bourdieu_docs = _get_continuum(
            self.embedding_model,
            new_docs,
            cont_name="cont2",
            left_words=self.bourdieu_query.y_top_words,
            right_words=self.bourdieu_query.y_bottom_words,
        )

        # Process and transform data
        df_bourdieu = pd.DataFrame(
            [
                {
                    "doc_id": x.doc_id,
                    "coordinates": [y.distance for y in x.bourdieu_dimensions],
                    "names": [y.continuum.id for y in x.bourdieu_dimensions],
                }
                for x in bourdieu_docs
            ]
        )
        df_bourdieu = df_bourdieu.explode(["coordinates", "names"])

        df_bourdieu_pivot = df_bourdieu[["doc_id", "coordinates", "names"]]
        df_bourdieu_pivot = df_bourdieu_pivot.pivot(
            index="doc_id", columns="names", values="coordinates"
        )

        # Add to the bourdieu_docs
        df_outsides = df_bourdieu_pivot.reset_index()
        df_outsides["cont1"] = df_outsides["cont1"].astype(float)
        df_outsides["cont2"] = df_outsides["cont2"].astype(float)

        x_values = df_outsides["cont1"].values
        y_values = df_outsides["cont2"].values

        distances = np.sqrt(x_values**2 + y_values**2)
        circle_radius = max(df_outsides.cont1) * self.bourdieu_query.radius_size

        df_outsides["distances"] = distances
        df_outsides["outside"] = "0"
        df_outsides.loc[df_outsides["distances"] >= circle_radius, "outside"] = "1"

        outside_ids = list(df_outsides["doc_id"][df_outsides["outside"] == "1"])
        bourdieu_docs = [x for x in bourdieu_docs if x.doc_id in outside_ids]
        bourdieu_dict = df_bourdieu_pivot.to_dict(orient="index")

        for doc in bourdieu_docs:
            doc.x = bourdieu_dict.get(doc.doc_id)["cont1"]
            doc.y = bourdieu_dict.get(doc.doc_id)["cont2"]

        # Compute Bourdieu topics
        topic_model = BunkaTopicModeling(
            n_clusters=self.topic_param.n_clusters,
            ngrams=self.topic_param.ngrams,
            name_length=self.topic_param.name_length,
            top_terms_overall=self.topic_param.top_terms_overall,
            min_count_terms=self.min_count_terms,
        )

        bourdieu_topics: t.List[Topic] = topic_model.fit_transform(
            docs=bourdieu_docs,
            terms=terms,
        )
        model_ranker = DocumentRanker(ranking_terms=self.ranking_terms)
        bourdieu_docs, bourdieu_topics = model_ranker.fit_transform(
            bourdieu_docs, bourdieu_topics
        )

        if self.llm:
            model_cleaning = LLMCleaningTopic(
                self.llm,
                language=self.topic_gen_param.language,
                use_doc=self.topic_gen_param.use_doc,
                context=self.topic_gen_param.context,
            )
            bourdieu_topics: t.List[Topic] = model_cleaning.fit_transform(
                bourdieu_topics,
                bourdieu_docs,
            )

        return bourdieu_docs, bourdieu_topics


def _get_continuum(
    embedding_model,
    docs: t.List[Document],
    cont_name: str = "emotion",
    left_words: list = ["hate"],
    right_words: list = ["love"],
    scale: bool = False,
) -> t.List[Document]:
    """
    Compute the Bourdieu continuum dimensions for a list of documents.

    Args:
        embedding_model: The embedding model.
        docs: List of documents.
        cont_name: Name of the continuum dimension.
        left_words: List of words representing the left side of the continuum.
        right_words: List of words representing the right side of the continuum.
        scale: Whether to scale the continuum distances.

    Returns:
        List of documents with Bourdieu dimensions.
    """
    # Create a DataFrame from the input documents
    df_docs = pd.DataFrame.from_records([doc.model_dump() for doc in docs])
    df_emb = df_docs[["doc_id", "embedding"]]
    df_emb = df_emb.set_index("doc_id")
    df_emb = pd.DataFrame(list(df_emb["embedding"]))
    df_emb.index = df_docs["doc_id"]

    continuum = ContinuumDimension(
        id=cont_name, left_words=left_words, right_words=right_words
    )

    # Compute the extremity embeddings
    left_embedding = embedding_model.embed_documents(continuum.left_words)
    right_embedding = embedding_model.embed_documents(continuum.right_words)

    left_embedding = pd.DataFrame(left_embedding).mean().values.reshape(1, -1)
    right_embedding = pd.DataFrame(right_embedding).mean().values.reshape(1, -1)

    # Compute the continuum embedding
    continuum_embedding = left_embedding - right_embedding
    df_continuum = pd.DataFrame(continuum_embedding)
    df_continuum.index = ["distance"]

    # Compute the Cosine Similarity
    full_emb = pd.concat([df_emb, df_continuum])
    df_bert = pd.DataFrame(cosine_similarity(full_emb))

    df_bert.index = full_emb.index
    df_bert.columns = full_emb.index
    df_bert = df_bert.iloc[-1:,].T
    df_bert = df_bert.sort_values("distance", ascending=False).reset_index()
    df_bert = df_bert[1:]
    df_bert = df_bert.rename(columns={"index": "doc_id"})
    final_df = pd.merge(df_bert, df_docs[["doc_id", "content"]], on="doc_id")

    if scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        final_df[["distance"]] = scaler.fit_transform(final_df[["distance"]])

    final_df = final_df.set_index("doc_id")
    final_df = final_df[["distance"]]

    distance_dict = final_df.to_dict("index")
    bourdieu_docs = docs.copy()
    for doc in bourdieu_docs:
        res = BourdieuDimension(
            continuum=continuum, distance=distance_dict.get(doc.doc_id)["distance"]
        )
        doc.bourdieu_dimensions.append(res)

    return bourdieu_docs
