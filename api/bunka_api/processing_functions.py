import os
import typing as t

from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings

from api.bunka_api.datamodel import TopicsResponse, TopicParameterApi
from bunkatopics.functions.bourdieu_api import bourdieu_api
from bunkatopics import Bunka
from bunkatopics.datamodel import (
    TopicGenParam,
)

open_ai_generative_model = OpenAI(
    openai_api_key=os.getenv("OPEN_AI_KEY"),
    openai_organization=os.getenv("OPEN_AI_ORG_ID"),
)
english_bunka = Bunka(
    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    language = 'en_core_web_sm'
)
french_bunka = Bunka(
    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    language = 'fr_core_news_lg'
)


def process_topics(
    full_docs: t.List[str], params: TopicParameterApi
):
    if params.language == "french":
        bunka = french_bunka
    else:
        bunka = english_bunka

    bunka.fit(full_docs)
    bunka.get_topics(
        n_clusters=params.n_clusters,
        name_lenght=params.name_lenght,
        min_count_terms=params.min_count_terms
    )
    if params.clean_topics:
        bunka.get_clean_topic_name(
            generative_model=open_ai_generative_model,
            language=language)

    docs = bunka.docs
    topics = bunka.topics

    return TopicsResponse(docs=docs, topics=topics)


def process_bourdieu(
    full_docs: t.List[str],
    bourdieu_query: BourdieuQueryApi,
    topic_param: TopicParameterApi
):
    if params.language == "french":
        bunka = french_bunka
    else:
        bunka = english_bunka

    bunka.fit(full_docs)
    bunka.get_topics(
        n_clusters=params.n_clusters,
        name_lenght=params.name_lenght,
        min_count_terms=params.min_count_terms
    )
    if params.clean_topics:
        bunka.get_clean_topic_name(
            generative_model=open_ai_generative_model,
            language=language)

    return bourdieu_api(
        generative_model=open_ai_generative_model,
        embedding_model=bunka.embedding_model,
        docs=bunka.docs,
        terms=bunka.terms,
        bourdieu_query=bourdieu_query,
        topic_param=topic_param,
        generative_ai_name=clean_topic,
        min_count_terms=1,
        topic_gen_param=TopicGenParam(),
    )
