import sys
from pprint import pprint

from dotenv import load_dotenv

load_dotenv()
sys.path.append("../")

import asyncio
import json
import typing as t

import pandas as pd
from celery.result import AsyncResult
from fastapi import Form, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse

from api.bunka_api.app import app
from api.bunka_api.datamodel import (
    BourdieuQueryApi,
    BourdieuQueryDict,
    BourdieuResponse,
    Document,
    Term,
    TopicParameterApi,
)
from api.bunka_api.jobs import bourdieu_api_task, process_topics_task

# Import the necessary modules and classes
from api.bunka_api.processing_functions import process_partial_bourdieu


def limit_docs(full_docs):
    if len(full_docs) > 10000:
        content = {
            "status_code": 10422,
            "message": "CSV must contain less than 10 000 lines",
            "data": None,
        }
        return JSONResponse(
            content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    else:
        return None


@app.post("/topics/")
def post_process_topics(full_docs: t.List[str], topic_param: TopicParameterApi):
    limit = limit_docs(full_docs)
    if limit is not None:
        return limit
    task = process_topics_task.delay(full_docs, topic_param.to_dict())
    return {"task_id": task.id}


@app.post("/topics/csv/")
async def upload_process_topics_csv(
    file: UploadFile,
    n_clusters: int = Form(...),
    language: str = Form(...),
    name_length: int = Form(...),
    clean_topics: bool = Form(...),
    min_count_terms: int = Form(...),
    selected_column: str = Form(...),
    process_bourdieu: bool = Form(...),
    x_left_words: str = Form(...),
    x_right_words: str = Form(...),
    y_top_words: str = Form(...),
    y_bottom_words: str = Form(...),
    radius_size: float = Form(...),
):
    # Read the CSV file
    df = pd.read_csv(file.file)
    full_docs = df[selected_column].tolist()
    limit = limit_docs(full_docs)
    if limit is not None:
        return limit

    topic_param = TopicParameterApi(
        n_clusters=n_clusters,
        language=language,
        name_lenght=name_length,
        clean_topics=clean_topics,
        min_count_terms=min_count_terms,
    )
    query = {}
    if process_bourdieu:
        query = BourdieuQueryApi(
            x_left_words=[x.strip() for x in x_left_words.split(",")],
            x_right_words=[x.strip() for x in x_right_words.split(",")],
            y_top_words=[x.strip() for x in y_top_words.split(",")],
            y_bottom_words=[x.strip() for x in y_bottom_words.split(",")],
            radius_size=radius_size,
        )
    task = process_topics_task.delay(
        full_docs,
        topic_param.to_dict(),
        process_bourdieu=process_bourdieu,
        bourdieu_query=query.to_dict(),
    )

    return {"task_id": task.id}


@app.post("/bourdieu/")
def post_process_bourdieu_query(
    full_docs: t.List[str], query: BourdieuQueryApi, topic_param: TopicParameterApi
):
    limit = limit_docs(full_docs)
    if limit is not None:
        return limit

    task = bourdieu_api_task.delay(full_docs, query.to_dict(), topic_param.to_dict())
    return {"task_id": task.id}


@app.post("/bourdieu/csv/")
async def upload_process_bourdieu_csv(
    file: UploadFile,
    n_clusters: int = Form(...),
    language: str = Form(...),
    name_length: int = Form(...),
    clean_topics: bool = Form(...),
    min_count_terms: int = Form(...),
    selected_column: str = Form(...),
    x_left_words: str = Form(...),
    x_right_words: str = Form(...),
    y_top_words: str = Form(...),
    y_bottom_words: str = Form(...),
    radius_size: float = Form(...),
):
    """Deprecated"""
    df = pd.read_csv(file.file)
    full_docs = df[selected_column].tolist()
    limit = limit_docs(full_docs)
    if limit is not None:
        return limit

    topic_param = TopicParameterApi(
        n_clusters=n_clusters,
        language=language,
        name_lenght=name_length,
        clean_topics=clean_topics,
        min_count_terms=min_count_terms,
    )
    query = BourdieuQueryApi(
        x_left_words=[x.strip() for x in x_left_words.split(",")],
        x_right_words=[x.strip() for x in x_right_words.split(",")],
        y_top_words=[x.strip() for x in y_top_words.split(",")],
        y_bottom_words=[x.strip() for x in y_bottom_words.split(",")],
        radius_size=radius_size,
    )
    task = bourdieu_api_task.delay(full_docs, query.to_dict(), topic_param.to_dict())
    return {"task_id": task.id}


def sse_format(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/tasks/{task_name}/{task_id}/progress")
async def get_task_progress(task_name: str, task_id: str):
    """
    Return a stream of state progress
    Client-side JS can use EventSource
    """
    if task_name == "topics":
        task = process_topics_task.AsyncResult(task_id)
    elif task_name == "bourdieu":
        task = bourdieu_api_task.AsyncResult(task_id)
    else:
        return JSONResponse(
            status_code=404,
            content={"message": "No result available or task not in success state."},
        )
    task = AsyncResult(task_id)

    async def event_stream():
        while not task.ready():
            if task.state == "PENDING":
                data = {"state": task.state, "progress": 0}
            elif task.state != "FAILURE":
                data = {
                    "state": task.state,
                    "progress": task.info.get("progress", 0) if task.info else 0,
                }
            else:
                data = {"state": task.state, "error": str(task.info)}  # Exception info
            yield sse_format(data)
            await asyncio.sleep(1)  # Sleep to avoid too frequent updates

        # Send final task result
        if task.state == "SUCCESS":
            yield sse_format({"state": task.state, "result": task.result})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/tasks/{task_name}/{task_id}/result")
async def get_task_result(task_name: str, task_id: str):
    """Return the result data"""
    if task_name == "topics":
        task = process_topics_task.AsyncResult(task_id)
    elif task_name == "bourdieu":
        task = bourdieu_api_task.AsyncResult(task_id)
    else:
        return JSONResponse(
            status_code=404,
            content={"message": "No result available or task not in success state."},
        )

    if task.state == "SUCCESS":
        return task.result
    else:
        return JSONResponse(
            status_code=404,
            content={"message": "No result available or task not in success state."},
        )


@app.post("/bourdieu/refresh/{task_id}")
def post_refresh_bourdieu_query(
    bourdieu_query: BourdieuQueryApi,
    topic_param: TopicParameterApi,
    task_id: str,
):
    task = process_topics_task.AsyncResult(task_id)
    if task.state == "SUCCESS":
        result = task.result
        docs = result["docs"]
        limit = limit_docs(docs)
        if limit is not None:
            return limit
        terms = result["terms"]

        res = process_partial_bourdieu(
            docs=[Document(**doc) for doc in docs],
            terms=[Term(**term) for term in terms],
            bourdieu_query=bourdieu_query,
            topic_param=topic_param,
        )

        # Extract the results
        bourdieu_docs = res[0]
        bourdieu_topics = res[1]

        return BourdieuResponse(
            docs=bourdieu_docs,
            topics=bourdieu_topics,
            query=BourdieuQueryDict(**bourdieu_query.to_dict()),
        )
    else:
        return JSONResponse(
            status_code=404,
            content={"message": "No result available or task not in success state."},
        )
