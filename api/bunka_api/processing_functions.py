import os
import typing as t

from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings

from api.bunka_api.datamodel import TopicsResponse, TopicParameterApi
from bunkatopics.functions.bourdieu_api import bourdieu_api
from bunkatopics import Bunka

open_ai_generative_model = OpenAI(
    openai_api_key=os.getenv("OPEN_AI_KEY"),
    openai_organization=os.getenv("OPEN_AI_ORG_ID"),
)
existing_bunka = Bunka(
    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)


def process_topics(
    full_docs: t.List[str], params: TopicParameterApi, clean_topic=False
):
    existing_bunka.fit(full_docs)
    existing_bunka.get_topics(
        n_clusters=params.n_clusters, name_lenght=1, min_count_terms=2
    )
    if clean_topic:
        existing_bunka.get_clean_topic_name(generative_model=open_ai_generative_model)
    docs = existing_bunka.docs
    topics = existing_bunka.topics

    return TopicsResponse(docs=docs, topics=topics)


def process_bourdieu(
    full_docs: t.List[str], bourdieu_query, topics_param, clean_topic=False
):
    bunka_response = process_topics(full_docs, TopicParameterApi(), clean_topic)
    return bourdieu_api(
        generative_model=open_ai_generative_model,
        embedding_model=existing_bunka.embedding_model,
        docs=bunka_response.docs,
        terms=bunka_response.terms,
        bourdieu_query=bourdieu_query,
        topic_param=topics_param,
        generative_ai_name=clean_topic,
        min_count_terms=2,
        topic_gen_param=TopicGenParam(generative_model=open_ai_generative_model),
    )
