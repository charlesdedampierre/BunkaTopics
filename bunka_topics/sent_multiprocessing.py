import spacy
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

nlp = spacy.load("fr_core_news_lg")


def get_sent(text):
    doc = nlp(text)
    res = [sentence.text for sentence in doc.sents]
    return res


def sent_multiprocessing(docs):
    with Pool(8) as p:
        res = list(tqdm(p.imap(get_sent, docs), total=len(docs)))

    return res


def sent_extract(docs):
    res = list(tqdm(map(get_sent, docs), total=len(docs)))

    return res


if __name__ == "__main__":

    # Main Algorithm
    res = sent_multiprocessing(docs)

    indexes = data["id"].to_list()
    final_dict = {"sentences": res, "doc_index": indexes}
