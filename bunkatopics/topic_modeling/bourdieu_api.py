import typing as t

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from bunkatopics.datamodel import (BourdieuDimension, BourdieuQuery,
                                   ContinuumDimension, Document, Term, Topic,
                                   TopicGenParam, TopicParam)
from bunkatopics.topic_modeling.document_topic_analyzer import \
    get_top_documents
from bunkatopics.topic_modeling.llm_topic_representation import \
    get_clean_topic_all
from bunkatopics.topic_modeling.topic_model_builder import get_topics

pd.options.mode.chained_assignment = None


def bourdieu_api(
    generative_model,
    embedding_model,
    docs: t.List[Document],
    terms: t.List[Term],
    bourdieu_query: BourdieuQuery = BourdieuQuery(),
    topic_param: TopicParam = TopicParam(),
    generative_ai_name: bool = False,
    topic_gen_param: TopicGenParam = TopicGenParam(),
    min_count_terms: int = 2,
) -> t.Tuple[t.List[Document], t.List[Topic]]:
    """
    Compute Bourdieu dimensions and topics for a list of documents.

    Args:
        generative_model: The generative AI model.
        embedding_model: The embedding model.
        docs: List of documents.
        terms: List of terms.
        bourdieu_query: BourdieuQuery object.
        topic_param: TopicParam object.
        generative_ai_name: Whether to generate AI-generated topic names.
        topic_gen_param: TopicGenParam object.
        min_count_terms: Minimum term count.

    Returns:
        Tuple of lists containing processed documents and topics.
    """
    # Reset Bourdieu dimensions for all documents
    for doc in docs:
        doc.bourdieu_dimensions = []

    # Compute Continuums
    new_docs = get_continuum(
        embedding_model,
        docs,
        cont_name="cont1",
        left_words=bourdieu_query.x_left_words,
        right_words=bourdieu_query.x_right_words,
    )
    bourdieu_docs = get_continuum(
        embedding_model,
        new_docs,
        cont_name="cont2",
        left_words=bourdieu_query.y_top_words,
        right_words=bourdieu_query.y_bottom_words,
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
    circle_radius = max(df_outsides.cont1) * bourdieu_query.radius_size

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
    bourdieu_topics = get_topics(
        docs=bourdieu_docs,
        terms=terms,
        n_clusters=topic_param.n_clusters,
        ngrams=topic_param.ngrams,
        name_length=topic_param.name_length,
        top_terms_overall=topic_param.top_terms_overall,
        min_count_terms=min_count_terms,
    )

    bourdieu_docs, bourdieu_topics = get_top_documents(
        bourdieu_docs, bourdieu_topics, ranking_terms=20
    )

    if generative_ai_name:
        bourdieu_topics = get_clean_topic_all(
            generative_model,
            bourdieu_topics,
            bourdieu_docs,
            language=topic_gen_param.language,
            context=topic_gen_param.context,
            use_doc=topic_gen_param.use_doc,
        )

    return bourdieu_docs, bourdieu_topics


def get_continuum(
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
    df_docs = pd.DataFrame.from_records([doc.dict() for doc in docs])
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