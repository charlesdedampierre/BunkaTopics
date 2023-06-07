import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import typing as t
from ..datamodel import TERM_ID


def get_coherence(
    topics: t.List[t.List[TERM_ID]],
    texts: t.List[t.List[TERM_ID]],
    coherence_type="c_v",
):
    word2id = Dictionary(texts)
    topics_terms = [topic.term_id[:10] for topic in topics]
    cm = CoherenceModel(
        topics=topics_terms, texts=texts, coherence=coherence_type, dictionary=word2id
    )
    coherence_per_topic = cm.get_coherence_per_topic()
    coherence_mean = np.mean(coherence_per_topic)

    return coherence_mean
