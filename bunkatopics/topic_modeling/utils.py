import numpy as np
import pandas as pd


def specificity(
    df: pd.DataFrame, X: str, Y: str, Z: str, top_n: int = 50
) -> pd.DataFrame:
    """
    Calculate specificity between two categorical variables X and Y in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing data.
        X (str): The name of the first categorical variable.
        Y (str): The name of the second categorical variable.
        Z (str): The name of the column to be used as a weight (if None, default weight is 1).
        top_n (int): The number of top results to return.

    Returns:
        pd.DataFrame: A DataFrame containing specificity scores between X and Y.
    """
    if Z is None:
        Z = "count_values"
        df[Z] = 1
        group = df.groupby([X, Y]).agg(count_values=(Z, "sum")).reset_index()
        cont = group.pivot(index=X, columns=Y, values=Z).fillna(0).copy()
    else:
        group = df.groupby([X, Y])[Z].sum().reset_index()
        cont = group.pivot(index=X, columns=Y, values=Z).fillna(0).copy()

    tx = df[X].value_counts()
    ty = df[Y].value_counts()

    cont = cont.astype(int)

    tx_df = pd.DataFrame(tx)
    tx_df.columns = ["c"]
    ty_df = pd.DataFrame(ty)
    ty_df.columns = ["c"]

    # Total observed values
    n = group[Z].sum()

    # Matrix product using pd.T to pivot one of the two series.
    indep = tx_df.dot(ty_df.T) / n

    cont = cont.reindex(indep.columns, axis=1)
    cont = cont.reindex(indep.index, axis=0)

    # Contingency Matrix
    ecart = (cont - indep) ** 2 / indep
    chi2 = ecart.sum(axis=1)
    chi2 = chi2.sort_values(ascending=False)
    spec = ecart * np.sign(cont - indep)

    # Edge Table of X, Y, specificity measure
    edge = spec.unstack().reset_index()
    edge = edge.rename(columns={0: "specificity_score"})
    edge = edge.sort_values(
        by=[X, "specificity_score"], ascending=[True, False]
    ).reset_index(drop=True)
    edge = edge[edge["specificity_score"] > 0]
    edge = edge.groupby([X]).head(top_n)
    edge = edge.reset_index(drop=True)

    return edge
