import sys

sys.path.append("../")

from fastapi import FastAPI
import typing as t
from fastapi.middleware.cors import CORSMiddleware


# Import the necessary modules and classes
from langchain.embeddings import HuggingFaceEmbeddings
from bunkatopics import Bunka
from bunkatopics.datamodel import TopicParam, TopicGenParam, BourdieuQuery
from bunkatopics.functions.bourdieu_api import bourdieu_api
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
from api.bunka_api.datamodel import BourdieuResponse, BunkaResponse, TopicParameter

load_dotenv()

open_ai_generative_model = OpenAI(openai_api_key=os.getenv("OPEN_AI_KEY"))

app = FastAPI()

# Allow requests from all origins (not recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/topics/")
def process_topics(params: TopicParameter, full_docs: t.List[str]):
    # Initialize your embedding_model and Bunka instance here
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bunka = Bunka(embedding_model=embedding_model)
    bunka.fit(full_docs)
    bunka.get_topics(n_clusters=params.n_cluster, name_lenght=2, min_count_terms=5)
    bunka.get_clean_topic_name(generative_model=open_ai_generative_model)

    docs = bunka.docs
    topics = bunka.topics

    return BunkaResponse(docs=docs, topics=topics)


@app.post("/process_bourdieu_query/")
def process_bourdieu_query(query: BourdieuQuery, full_docs: t.List[str]):
    # Initialize your embedding_model and Bunka instance here
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bunka = Bunka(embedding_model=embedding_model)
    bunka.fit(full_docs)

    # Call your bourdieu_api function to get results
    res = bourdieu_api(
        bunka.embedding_model,
        bunka.docs,
        bunka.terms,
        bourdieu_query=query,
        topic_param=TopicParam(n_clusters=10),
        generative_ai_name=False,
        topic_gen_param=TopicGenParam(generative_model=open_ai_generative_model),
    )

    # Extract the results
    bourdieu_docs = res[0]
    bourdieu_topics = res[1]

    return BourdieuResponse(
        bourdieu_docs=bourdieu_docs, bourdieu_topics=bourdieu_topics
    )
