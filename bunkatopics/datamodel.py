import typing as t

from pydantic import BaseModel, Field

TOPIC_ID = str
TERM_ID = str
DOC_ID = str


class ContinuumDimension(BaseModel):
    id: str = "continuum"
    left_words: t.List[str] = Field(default_factory=list)
    right_words: t.List[str] = Field(default_factory=list)


class BourdieuDimension(BaseModel):
    continuum: ContinuumDimension
    distance: float


class BourdieuQuery(BaseModel):
    x_left_words: t.List[str] = ["war"]
    x_right_words: t.List[str] = ["peace"]
    y_top_words: t.List[str] = ["men"]
    y_bottom_words: t.List[str] = ["women"]
    radius_size: float = 0.5


class TopicRanking(BaseModel):
    topic_id: TOPIC_ID
    rank: int = None


class Document(BaseModel):
    doc_id: DOC_ID
    content: str
    size: t.Optional[float] = None
    x: t.Optional[float] = None
    y: t.Optional[float] = None
    topic_id: t.Optional[TOPIC_ID] = None
    topic_ranking: TopicRanking = None
    term_id: t.Optional[t.List[TERM_ID]] = None
    embedding: t.Optional[t.List[float]] = Field(None, repr=False)
    bourdieu_dimensions: t.List[BourdieuDimension] = []


class TopicParam(BaseModel):
    n_clusters = 5
    ngrams = [1, 2]
    name_lenght = 3
    top_terms_overall = 500


class TopicGenParam(BaseModel):
    language: str = "english"
    top_doc: int = 3
    top_terms: int = 10
    use_doc = False
    context: str = "everything"


class ConvexHullModel(BaseModel):
    # topic_id: TOPIC_ID
    x_coordinates: t.Optional[t.List[float]] = None
    y_coordinates: t.Optional[t.List[float]] = None


class Topic(BaseModel):
    topic_id: TOPIC_ID
    name: str
    lemma_name: t.Optional[str] = None
    term_id: t.List[TERM_ID] = Field(None, repr=False)
    x_centroid: t.Optional[float] = None
    y_centroid: t.Optional[float] = None
    size: t.Optional[int] = None
    top_doc_id: t.Optional[t.List[DOC_ID]] = None
    top_term_id: t.Optional[t.List[TERM_ID]] = None
    convex_hull: t.Optional[ConvexHullModel] = Field(None, repr=False)


class Term(BaseModel):
    term_id: str
    lemma: str
    ent: str
    ngrams: int
    count_terms: int
