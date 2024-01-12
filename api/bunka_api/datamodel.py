from pydantic import BaseModel
import typing as t

from bunkatopics.datamodel import Document, Topic, TopicParam, BourdieuQuery, Term


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

class BourdieuResponse(BaseModel):
    docs: t.List[Document]
    topics: t.List[Topic]
    query: BourdieuQueryDict


class TopicsResponse(BaseModel):
    docs: t.List[Document]
    topics: t.List[Topic]
    bourdieu_response: BourdieuResponse | None
    terms: t.List[Term]

class TopicParameterApi(TopicParam):
    """API specific Topics parameters"""

    n_clusters: int = 10
    language : str = "english"
    clean_topics = True
    min_count_terms: int = 1
    
    def to_dict(self):
        return {
            # Convert all attributes to a serializable format
            "n_clusters": self.n_clusters,
            "ngrams": self.ngrams,
            "name_lenght": self.name_lenght,
            "top_terms_overall": self.top_terms_overall,
            "language": self.language,
            "clean_topics": self.clean_topics,
            "min_count_terms": self.min_count_terms
        }
