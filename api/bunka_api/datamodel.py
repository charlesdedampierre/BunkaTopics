from pydantic import BaseModel
import typing as t

from bunkatopics.datamodel import Document, Topic, TopicParam


class BourdieuResponse(BaseModel):
    bourdieu_docs: t.List[Document]
    bourdieu_topics: t.List[Topic]


class BunkaResponse(BaseModel):
    docs: t.List[Document]
    topics: t.List[Topic]

class TopicParameter(TopicParam):
    """Override default value"""
    n_clusters = 10

    def to_dict(self):
        return {
            # Convert all attributes to a serializable format
            "n_clusters": self.n_clusters,
            "ngrams": self.ngrams,
            "name_lenght": self.name_lenght,
            "top_terms_overall": self.top_terms_overall
        }
