import os
import typing as t

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

from api.bunka_api.app import app
from api.bunka_api.datamodel import BunkaResponse, TopicParameterApi
from bunkatopics.functions.bourdieu_api import bourdieu_api
from bunkatopics import Bunka

open_ai_generative_model = OpenAI(openai_api_key=os.getenv("OPEN_AI_KEY"))


def get_or_init_bunka_instance():
    existing_bunka = app.state.existing_bunka.bunka

    if existing_bunka is None:
        existing_bunka = Bunka(
            embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        app.state.existing_bunka.bunka = existing_bunka

    return existing_bunka


def process_topics(full_docs: t.List[str], params: TopicParameterApi):
    existing_bunka = get_or_init_bunka_instance()
    existing_bunka.fit(full_docs)
    existing_bunka.get_topics(
        n_clusters=params.n_clusters, name_lenght=1, min_count_terms=2
    )
    existing_bunka.get_clean_topic_name(generative_model=open_ai_generative_model)
    docs = existing_bunka.docs
    topics = existing_bunka.topics

    return BunkaResponse(docs=docs, topics=topics)


def process_bourdieu(full_docs: t.List[str], bourdieu_query, topics_param):
    bunka_response = process_topics(full_docs, TopicParameterApi())
    existing_bunka = get_or_init_bunka_instance()
    return bourdieu_api(
        generative_model=open_ai_generative_model,
        embedding_model=existing_bunka.embedding_model,
        docs=bunka_response.docs,
        terms=bunka_response.terms,
        bourdieu_query=bourdieu_query,
        topic_param=topics_param,
        generative_ai_name=False,
        min_count_terms=2,
        topic_gen_param=TopicGenParam(generative_model=open_ai_generative_model),
    )
