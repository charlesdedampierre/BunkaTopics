from pydantic import BaseModel
import typing as t

from bunkatopics.datamodel import Document, Topic, TopicParam, BourdieuQuery


class BourdieuQueryDict(t.TypedDict):
    x_left_words: t.List[str]
    x_right_words: t.List[str]
    y_top_words: t.List[str]
    y_bottom_words: t.List[str]
    radius_size: float


class BourdieuQueryApi(BourdieuQuery):
    def to_dict(self):
        return {
            "x_left_words": self.x_left_words,
            "x_right_words": self.x_right_words,
            "y_top_words": self.y_top_words,
            "y_bottom_words": self.y_bottom_words,
            "radius_size": self.radius_size,
        }


class TopicsResponse(BaseModel):
    docs: t.List[Document]
    topics: t.List[Topic]


class BourdieuResponse(TopicsResponse):
    query: BourdieuQueryDict


class TopicParameterApi(TopicParam):
    """Override default value"""

    n_clusters = 10

    def to_dict(self):
        return {
            # Convert all attributes to a serializable format
            "n_clusters": self.n_clusters,
            "ngrams": self.ngrams,
            "name_lenght": self.name_lenght,
            "top_terms_overall": self.top_terms_overall,
        }
