import pandas as pd
from tqdm import tqdm


def indexer(list_to_index, docs):

    final = pd.DataFrame()
    for term in tqdm(list_to_index, total=len(list_to_index)):
        new = pd.DataFrame(docs, columns=["docs"])
        res = new[new["docs"].str.contains(term, case=False)]
        res["indexed_term"] = term
        final = final.append(res)

    final = final.drop_duplicates().reset_index(drop=True)

    return final
