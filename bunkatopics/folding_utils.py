import pandas as pd
from tqdm.notebook import tqdm

# Index
def index_data(vocabulary, docs):
    results = []
    for doc in tqdm(list(docs), total=len(docs)):
        for annot in set(vocabulary):
            res = doc.find(annot)
            if res != -1:
                results.append((doc, annot, res))
            else:
                pass

    final = pd.DataFrame(results, columns=["body", "sentence_annot", "sent_location"])
    return final
