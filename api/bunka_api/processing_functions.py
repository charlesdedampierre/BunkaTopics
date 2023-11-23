import os
import typing as t

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI

from api.bunka_api.app import app
from api.bunka_api.datamodel import BunkaResponse, TopicParameter
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


def process_topics(full_docs: t.List[str], params: TopicParameter):
    existing_bunka = get_or_init_bunka_instance()
    existing_bunka.fit(full_docs)
    existing_bunka.get_topics(
        n_clusters=params.n_clusters, name_lenght=1, min_count_terms=2
    )
    existing_bunka.get_clean_topic_name(generative_model=open_ai_generative_model)
    docs = existing_bunka.docs
    topics = existing_bunka.topics

    return BunkaResponse(docs=docs, topics=topics)


def process_bourdieu(open_ai_generative_model, existing_bunka, topics_param):
    existing_bunka = get_or_init_bunka_instance()

    return bourdieu_api(
        generative_model=open_ai_generative_model,
        embedding_model=existing_bunka.embedding_model,
        docs=existing_bunka.docs,
        terms=existing_bunka.terms,
        bourdieu_query=query,
        topic_param=topics_param,
        generative_ai_name=False,
        min_count_terms=2,
        topic_gen_param=TopicGenParam(generative_model=open_ai_generative_model),
    )
