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


@app.post("/topics_test/")
def process_topics(params: TopicParameter, full_docs: t.List[str]):
    # Initialize your embedding_model and Bunka instance here
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    bunka = Bunka(embedding_model=embedding_model)
    bunka.fit(full_docs)
    bunka.get_topics(n_clusters=params.n_cluster, name_lenght=3)

    docs = bunka.docs
    topics = bunka.topics

    return BunkaResponse(docs=docs, topics=topics)


@app.post("/topics/")
def post_process_topics(n_clusters, full_docs: t.List[str]):
    return process_topics(full_docs, n_clusters)


@app.post("/process_bourdieu_query/")
def post_process_bourdieu_query(query: BourdieuQuery, full_docs: t.List[str]):
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
