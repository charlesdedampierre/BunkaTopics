from pydantic import BaseModel
import typing as t

from bunkatopics.datamodel import Document, Topic


class BourdieuResponse(BaseModel):
    bourdieu_docs: t.List[Document]
    bourdieu_topics: t.List[Topic]
