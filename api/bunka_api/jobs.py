from celery import Celery, states
from celery.exceptions import Ignore
import typing as t

from api.bunka_api.processing_functions import (
    process_topics,
    process_bourdieu,
    open_ai_generative_model,
)
from api.bunka_api.datamodel import TopicParameterApi, BourdieuQueryApi
from api import celeryconfig

celery = Celery()
celery.config_from_object(celeryconfig)


@celery.task(bind=True)
def process_topics_task(self, full_docs: t.List[str], params: t.Dict):
    try:
        # Initialization
        total = len(full_docs)
        self.update_state(state=states.STARTED, meta={"progress": 0})
        result = process_topics(full_docs, TopicParameterApi(
            n_clusters=params.n_clusters
        ))
        # TODO get the real progress
        i = total
        progress = (i + 1) / total * 100
        self.update_state(state="PROCESSING", meta={"progress": progress})
        # Task completed
        return result

    except Exception as e:
        # Handle exceptions
        print(e)
        self.update_state(
            state=states.FAILURE,
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise Ignore()


@celery.task(bind=True)
def bourdieu_api_task(self, full_docs: t.List[str], bourdieu_query: t.Dict, topics_param: t.Dict):
    try:
        # Initialization
        total = len(full_docs)
        topics_param_ins =  TopicParameterApi(n_clusters=topics_param.n_clusters)
        bourdieu_query_ins = BourdieuQueryApi(
            x_left_words=bourdieu_query.x_left_words, 
            x_right_words=bourdieu_query.x_right_words, 
            y_top_words=bourdieu_query.y_top_words, 
            y_bottom_words=bourdieu_query.y_bottom_words, 
            radius_size=bourdieu_query.radius_size)
        self.update_state(state=states.STARTED, meta={"progress": 0})
        res = process_bourdieu(
            full_docs=full_docs,
            bourdieu_query=bourdieu_query_ins,
            topic_param=topicstopics_param_ins_param,
        )
        # TODO get the real progress
        i = total
        progress = (i + 1) / total * 100
        self.update_state(state="PROCESSING", meta={"progress": progress})
        # Extract the results
        bourdieu_docs = res[0]
        bourdieu_topics = res[1]

        return BourdieuResponse(
            bourdieu_docs=bourdieu_docs, bourdieu_topics=bourdieu_topics
        )

    except Exception as e:
        print(e)
        # Handle exceptions
        self.update_state(
            state=states.FAILURE,
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise Ignore()
