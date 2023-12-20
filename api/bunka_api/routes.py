import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append("../")

import json
import time
import typing as t
import os
import asyncio
from celery.result import AsyncResult
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import UploadFile, Form, status
import pandas as pd

# Import the necessary modules and classes
from api.bunka_api.app import app
from api.bunka_api.datamodel import (
    TopicParameterApi,
    BourdieuQueryApi,
)
from api.bunka_api.jobs import process_topics_task, bourdieu_api_task

def limit_docs(full_docs):
    if len(full_docs) > 10000:
        content = {"status_code": 10422, "message": "CSV must contain less than 10 000 lines", "data": None}
        return JSONResponse(
            content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    else:
        return None


@app.post("/topics/")
def post_process_topics(full_docs: t.List[str], topics_param: TopicParameterApi):
    limit = limit_docs(full_docs)
    if limit is not None:
        return limit
    task = process_topics_task.delay(full_docs, topics_param.to_dict())
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
):
    # Read the CSV file
    df = pd.read_csv(file.file)
    full_docs = df[selected_column].tolist()
    limit = limit_docs(full_docs)
    if limit is not None:
        return limit

    topics_param = TopicParameterApi(
        n_clusters=n_clusters,
        language=language,
        name_lenght=name_length,
        clean_topics=clean_topics,
        min_count_terms=min_count_terms
    )
    task = process_topics_task.delay(full_docs, topics_param.to_dict())
    return {"task_id": task.id}


@app.post("/bourdieu/")
def post_process_bourdieu_query(
    full_docs: t.List[str],
    query: BourdieuQueryApi,
    topics_param: TopicParameterApi
):
    limit = limit_docs(full_docs)
    if limit is not None:
        return limit

    task = bourdieu_api_task.delay(full_docs, query.to_dict(), topics_param.to_dict())
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
    # Read the CSV file
    df = pd.read_csv(file.file)
    full_docs = df[selected_column].tolist()
    limit = limit_docs(full_docs)
    if limit is not None:
        return limit

    topics_param = TopicParameterApi(
        n_clusters=n_clusters,
        language=language,
        name_lenght=name_length,
        clean_topics=clean_topics,
        min_count_terms=min_count_terms
    )
    query = BourdieuQueryApi(
        x_left_words=[x.strip() for x in x_left_words.split(",")],
        x_right_words=[x.strip() for x in x_right_words.split(",")],
        y_top_words=[x.strip() for x in y_top_words.split(",")],
        y_bottom_words=[x.strip() for x in y_bottom_words.split(",")],
        radius_size=radius_size,
    )
    task = bourdieu_api_task.delay(full_docs, query.to_dict(), topics_param.to_dict())
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


# @app.get("/tasks/{task_name}/{task_id}/result")
# async def get_task_result(task_name: str, task_id: str):
#     """Return the result data"""
#     if task_name == "topics":
#         task = process_topics_task.AsyncResult(task_id)
#     elif task_name == "bourdieu":
#         task = bourdieu_api_task.AsyncResult(task_id)
#     else:
#         return JSONResponse(
#             status_code=404,
#             content={"message": "No result available or task not in success state."},
#         )

#     if task.state == "SUCCESS":
#         return task.result
#     else:
#         return JSONResponse(
#             status_code=404,
#             content={"message": "No result available or task not in success state."},
#         )
