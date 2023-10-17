import sys

sys.path.append("../")

from fastapi import FastAPI
import typing as t

# Import the necessary modules and classes
from langchain.embeddings import HuggingFaceEmbeddings
from bunkatopics import Bunka
from bunkatopics.datamodel import (
    TopicParam,
    TopicGenParam,
    BourdieuQuery,
    Document,
    Topic,
)
from bunkatopics.functions.bourdieu_api import bourdieu_api
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
from api.datamodel import BourdieuResponse

load_dotenv()

open_ai_generative_model = OpenAI(openai_api_key=os.getenv("OPEN_AI_KEY"))

app = FastAPI()


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
