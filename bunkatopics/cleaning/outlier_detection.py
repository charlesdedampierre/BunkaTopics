import numpy as np
import pandas as pd
from pyod.models.ecod import ECOD


def remove_outliers(docs, threshold=6):
    df_2 = pd.DataFrame(
        {
            "doc_id": [x.doc_id for x in docs],
            "x": [x.x for x in docs],
            "y": [x.y for x in docs],
        }
    )
    clf = ECOD()
    X_train = df_2[["x", "y"]].to_numpy()
    clf.fit(X_train)
    y_train_scores = clf.decision_scores_  # Outlier scores for training data
    outlier_indices = np.where(y_train_scores > threshold)[0]
    df_filtered = df_2.drop(outlier_indices)
    list_index = list(df_filtered["doc_id"])

    filtered_docs = [doc for doc in docs if doc.doc_id in list_index]
    return filtered_docs
