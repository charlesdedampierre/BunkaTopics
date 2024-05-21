import logging
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.append("../")

import typing as t

from celery import Celery, states
from celery.exceptions import Ignore
from fastapi.encoders import jsonable_encoder

from api import celeryconfig
from api.bunka_api.datamodel import (
    BourdieuQueryApi,
    BourdieuQueryDict,
    BourdieuResponse,
    TopicParameterApi,
)
from api.bunka_api.processing_functions import (
    process_full_topics_and_bourdieu,
    process_topics,
)

celery = Celery()
celery.config_from_object(celeryconfig)


@celery.task(bind=True)
def process_topics_task(
    self,
    full_docs: t.List[str],
    params: t.Dict,
    process_bourdieu: bool,
    bourdieu_query: t.Dict,
):
    try:
        # Initialization
        total = len(full_docs)
        self.update_state(state=states.STARTED, meta={"progress": 0})
        query = None
        if process_bourdieu:
            query = BourdieuQueryApi(
                x_left_words=bourdieu_query["x_left_words"],
                x_right_words=bourdieu_query["x_right_words"],
                y_top_words=bourdieu_query["y_top_words"],
                y_bottom_words=bourdieu_query["y_bottom_words"],
                radius_size=bourdieu_query["radius_size"],
            )

        result = process_topics(
            full_docs,
            TopicParameterApi(
                n_clusters=params["n_clusters"],
                language=params["language"],
                clean_topics=params["clean_topics"],
                min_count_terms=params["min_count_terms"],
                name_lenght=params["name_lenght"],
            ),
            process_bourdieu,
            bourdieu_query=query,
        )
        # TODO get the real progress
        i = total
        progress = (i + 1) / total * 100
        self.update_state(state="PROCESSING", meta={"progress": progress})
        # Task completed
        return jsonable_encoder(result)

    except Exception as e:
        # Handle exceptions
        logging.error(e)
        self.update_state(
            state=states.FAILURE,
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise Ignore


@celery.task(bind=True)
def bourdieu_api_task(
    self, full_docs: t.List[str], bourdieu_query: t.Dict, topic_param: t.Dict
):
    """Deprecated"""
    try:
        # Initialization
        total = len(full_docs)
        topics_param_ins = TopicParameterApi(
            n_clusters=topic_param["n_clusters"],
            language=topic_param["language"],
            clean_topics=topic_param["clean_topics"],
            min_count_terms=topic_param["min_count_terms"],
            name_lenght=topic_param["name_lenght"],
        )
        bourdieu_query_ins = BourdieuQueryApi(
            x_left_words=bourdieu_query["x_left_words"],
            x_right_words=bourdieu_query["x_right_words"],
            y_top_words=bourdieu_query["y_top_words"],
            y_bottom_words=bourdieu_query["y_bottom_words"],
            radius_size=bourdieu_query["radius_size"],
        )
        self.update_state(state=states.STARTED, meta={"progress": 0})
        res = process_full_topics_and_bourdieu(
            full_docs=full_docs,
            bourdieu_query=bourdieu_query_ins,
            topic_param=topics_param_ins,
        )
        # TODO get the real progress
        i = total
        progress = (i + 1) / total * 100
        self.update_state(state="PROCESSING", meta={"progress": progress})
        # Extract the results
        bourdieu_docs = res[0]
        bourdieu_topics = res[1]

        response = BourdieuResponse(
            docs=bourdieu_docs,
            topics=bourdieu_topics,
            query=BourdieuQueryDict(**bourdieu_query_ins.to_dict()),
        )
        return jsonable_encoder(response)

    except Exception as e:
        logging.error(e)
        # Handle exceptions
        self.update_state(
            state=states.FAILURE,
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise Ignore()
