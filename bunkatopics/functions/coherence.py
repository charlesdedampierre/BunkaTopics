import typing as t

import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel

from ..datamodel import TERM_ID


def get_coherence(
    topics: t.List[t.List[TERM_ID]],
    texts: t.List[t.List[TERM_ID]],
    coherence_type="c_v",
    topic_terms_n: int = 10,
):
    word2id = Dictionary(texts)
    # topics_terms = [topic.term_id[:topic_terms_n] for topic in topics]
    # topics_terms = [topic.name.split(" | ")[:topic_terms_n] for topic in topics]
    topics_terms = [x.term_id[:topic_terms_n] for x in topics]

    cm = CoherenceModel(
        topics=topics_terms, texts=texts, coherence=coherence_type, dictionary=word2id
    )
    coherence_per_topic = cm.get_coherence_per_topic()
    coherence_mean = np.mean(coherence_per_topic)

    return coherence_mean
