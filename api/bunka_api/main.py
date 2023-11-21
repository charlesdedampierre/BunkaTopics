import sys

sys.path.append("../")

import typing as t
import pandas as pd
import logging
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, Form, Request, status
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


# Allow requests from all origins (not recommended for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/topics/")
def post_process_topics(n_clusters, full_docs: t.List[str]):
    return process_topics(full_docs, n_clusters)


class GlobalBunka:
    def __init__(self):
        self.bunka = None
        self.bunka_results = None


# Initialize a global instance during startup
@app.on_event("startup")
def startup_event():
    app.state.existing_bunka = GlobalBunka()


# Your /topics_test/ endpoint
@app.post("/topics_test/")
def process_topics(params: TopicParameter, full_docs: t.List[str]):
    existing_bunka = app.state.existing_bunka.bunka

    if existing_bunka is None:
        existing_bunka = Bunka(
            embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        app.state.existing_bunka.bunka = existing_bunka

    existing_bunka.fit(full_docs)
    existing_bunka.get_topics(
        n_clusters=params.n_cluster, name_lenght=1, min_count_terms=2
    )

    docs = existing_bunka.docs
    topics = existing_bunka.topics

    return BunkaResponse(docs=docs, topics=topics)


@app.post("/bourdieu/")
def post_process_bourdieu_query(query: BourdieuQuery, topics_param: TopicParam):
    existing_bunka = app.state.existing_bunka.bunka

    if existing_bunka is None:
        # Handle the case where /topics_test/ hasn't been run yet
        return {"error": "Run /topics_test/ first"}

    # Call your bourdieu_api function to get results
    res = bourdieu_api(
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

    # Extract the results
    bourdieu_docs = res[0]
    bourdieu_topics = res[1]

    return BourdieuResponse(
        bourdieu_docs=bourdieu_docs, bourdieu_topics=bourdieu_topics
    )
