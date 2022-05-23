import pandas as pd
import numpy as np


def specificity(df: pd.DataFrame, X: str, Y: str, Z: str, top_n: int = 50):

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

    # Valeurs totales observées
    n = group[Z].sum()

    # Produit matriciel. On utilise pd.T pour pivoter une des deux séries.
    indep = tx_df.dot(ty_df.T) / n

    cont = cont.reindex(indep.columns, axis=1)
    cont = cont.reindex(indep.index, axis=0)

    # Contingency Matrix
    ecart = (cont - indep) ** 2 / indep
    chi2 = ecart.sum(axis=1)
    chi2 = chi2.sort_values(ascending=False)
    spec = ecart * np.sign(cont - indep)

    # Edge Table of X, Y, specificity measure
    spec[X] = spec.index
    edge = pd.melt(spec, id_vars=[X])
    edge.columns = [X, Y, "spec"]
    edge = edge.sort_values(by=[X, "spec"], ascending=[True, False]).reset_index(
        drop=True
    )
    edge = edge[edge.spec > 0]
    edge = edge.groupby([X]).head(top_n)
    edge = edge.reset_index(drop=True)

    return spec, chi2, edge
