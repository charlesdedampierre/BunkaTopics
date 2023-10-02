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


class Document(BaseModel):
    doc_id: DOC_ID
    content: str
    size: t.Optional[float] = None
    x: t.Optional[float] = None
    y: t.Optional[float] = None
    topic_id: t.Optional[TOPIC_ID] = None
    term_id: t.Optional[t.List[TERM_ID]] = None
    embedding: t.Optional[t.List[float]] = Field(None, repr=False)
    bourdieu_dimensions: t.List[BourdieuDimension] = []


class ConvexHullModel(BaseModel):
    topic_id: TOPIC_ID
    x_coordinates: t.Optional[t.List[float]] = None
    y_coordinates: t.Optional[t.List[float]] = None


class Topic(BaseModel):
    topic_id: TOPIC_ID
    name: str
    lemma_name: t.Optional[str] = None
    term_id: t.List[TERM_ID]
    x_centroid: t.Optional[float] = None
    y_centroid: t.Optional[float] = None
    size: t.Optional[int] = None
    top_doc_id: t.Optional[t.List[DOC_ID]] = None
    top_term_id: t.Optional[t.List[TERM_ID]] = None


class Term(BaseModel):
    term_id: str
    lemma: str
    ent: str
    ngrams: int
    count_terms: int
