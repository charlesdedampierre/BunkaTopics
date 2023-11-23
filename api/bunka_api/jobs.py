from celery import Celery, states
from celery.exceptions import Ignore
import typing as t

from bunkatopics.datamodel import BourdieuQuery
from api.bunka_api.datamodel import TopicParameter
from api.bunka_api.processing_functions import (
    process_topics,
    process_bourdieu,
    open_ai_generative_model,
)

from api import celeryconfig

celery = Celery()
celery.config_from_object(celeryconfig)


@celery.task(bind=True)
def process_topics_task(self, full_docs: t.List[str], params: t.Dict):
    try:
        # Initialization
        total = len(full_docs)
        self.update_state(state=states.STARTED, meta={"progress": 0})
        result = process_topics(full_docs, params)
        # TODO get the real progress
        i = total
        progress = (i + 1) / total * 100
        self.update_state(state="PROCESSING", meta={"progress": progress})
        # Task completed
        return result

    except Exception as e:
        # Handle exceptions
        self.update_state(
            state=states.FAILURE,
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise Ignore()


@celery.task(bind=True)
def bourdieu_api_task(self, query: BourdieuQuery, topics_param: t.Dict):
    try:
        # Initialization
        total = len(full_docs)
        self.update_state(state=states.STARTED, meta={"progress": 0})
        res = process_bourdieu(
            generative_model=open_ai_generative_model,
            bunka_instance=existing_bunka,
            bourdieu_query=query,
            topic_param=topics_param,
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
        # Handle exceptions
        self.update_state(
            state=states.FAILURE,
            meta={"exc_type": type(e).__name__, "exc_message": str(e)},
        )
        raise Ignore()
