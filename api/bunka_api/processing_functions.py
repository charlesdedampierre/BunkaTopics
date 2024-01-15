import copy
import os
import typing as t

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

from api.bunka_api.datamodel import (BourdieuQueryApi, BourdieuQueryDict,
                                     BourdieuResponse, TopicParameterApi,
                                     TopicsResponse)
from bunkatopics import Bunka
from bunkatopics.datamodel import Document, Term, Topic, TopicGenParam
from bunkatopics.functions.bourdieu_api import bourdieu_api

open_ai_generative_model = OpenAI(
    openai_api_key=os.getenv("OPEN_AI_KEY"),
    openai_organization=os.getenv("OPEN_AI_ORG_ID")
    #    model=
    #    client=
)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
english_bunka_language = "en_core_web_sm"
french_bunka_language = "fr_core_news_lg"


def process_topics(
    full_docs: t.List[str],
    params: TopicParameterApi,
    process_bourdieu: bool,
    bourdieu_query: BourdieuQueryApi | None,
) -> TopicsResponse:
    """Process topics and bourdieu query if asked for"""
    if params.language == "french":
        bunka = Bunka(embedding_model=embedding_model, language=french_bunka_language)
    else:
        bunka = Bunka(embedding_model=embedding_model, language=english_bunka_language)

    bunka.fit(full_docs)
    bunka.get_topics(
        n_clusters=params.n_clusters,
        name_lenght=params.name_lenght,
        min_count_terms=params.min_count_terms,
    )
    if params.clean_topics:
        bunka.get_clean_topic_name(
            generative_model=open_ai_generative_model, language=params.language
        )

    bourdieu_response = None

    if process_bourdieu and bourdieu_query is not None:
        (bourdieu_docs, bourdieu_topics) = process_partial_bourdieu(
            docs=bunka.docs,
            terms=bunka.terms,
            bourdieu_query=bourdieu_query,
            topic_param=params,
        )
        query_dict = BourdieuQueryDict(**bourdieu_query.to_dict())
        bourdieu_response = BourdieuResponse(
            docs=bourdieu_docs, topics=bourdieu_topics, query=query_dict
        )

    return TopicsResponse(
        docs=bunka.docs,
        topics=bunka.topics,
        bourdieu_response=bourdieu_response,
        # Store terms to reprocess bourdieu later
        terms=bunka.terms,
    )


def process_full_topics_and_bourdieu(
    full_docs: t.List[str],
    bourdieu_query: BourdieuQueryApi,
    topic_param: TopicParameterApi,
):
    """
    Deprecated
    Full process of topics then the Bourdieu view
    """
    if topic_param.language == "french":
        bunka = Bunka(embedding_model=embedding_model, language=french_bunka_language)
    else:
        bunka = Bunka(embedding_model=embedding_model, language=english_bunka_language)

    bunka.fit(full_docs)
    bunka.get_topics(
        n_clusters=topic_param.n_clusters,
        name_lenght=topic_param.name_lenght,
        min_count_terms=topic_param.min_count_terms,
    )
    if topic_param.clean_topics:
        bunka.get_clean_topic_name(
            generative_model=open_ai_generative_model, language=topic_param.language
        )

    return bourdieu_api(
        generative_model=open_ai_generative_model,
        embedding_model=bunka.embedding_model,
        docs=bunka.docs,
        terms=bunka.terms,
        bourdieu_query=bourdieu_query,
        topic_param=topic_param,
        generative_ai_name=topic_param.clean_topics,
        min_count_terms=1,
        topic_gen_param=TopicGenParam(),
    )


def process_partial_bourdieu(
    docs: t.List[Document],
    terms: t.List[Term],
    bourdieu_query: BourdieuQueryApi,
    topic_param: TopicParameterApi,
) -> t.Tuple[t.List[Document], t.List[Topic]]:
    """Process a bourdieu view with topics already processed into the bunka instance"""

    return bourdieu_api(
        generative_model=open_ai_generative_model,
        embedding_model=embedding_model,
        docs=copy.deepcopy(docs),  # Make a copy of the variable
        terms=copy.deepcopy(terms),
        bourdieu_query=bourdieu_query,
        topic_param=topic_param,
        generative_ai_name=topic_param.clean_topics,
        min_count_terms=1,  # TODO parametrize ?
        topic_gen_param=TopicGenParam(language=topic_param.language),
    )
