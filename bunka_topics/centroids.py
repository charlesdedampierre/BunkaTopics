import numpy as np
import pandas as pd


def closest_point(centroid, matrix, n=2):
    # Get the n closest points to the centroids
    matrix = np.asarray(matrix)
    dist_2 = np.sum((matrix - centroid) ** 2, axis=1)
    point = np.argsort(dist_2)[:n]
    return point


def farest_point(centroid, matrix):
    matrix = np.asarray(matrix)
    dist_2 = np.sum((matrix - centroid) ** 2, axis=1)
    point = np.argsort(dist_2)[-1]
    # point = np.argmax(dist_2)
    return point


def find_centroids(
    df: pd.DataFrame,
    text_var,
    cluster_var="clusters",
    top_elements: int = 4,
    dim_lenght: int = 5,
) -> pd.DataFrame:
    """Find the top centroids among clusters

    Parameters
    ----------
    df : pd.DataFrame
        with following columns:
            text_var, cluster_var,0,1,2,3,4 for a 4 dimension embeddings

    Returns
    -------
    _type_
        _description_
    """

    clusters = []
    centroid_elements = []
    far_elements = []
    distances = []

    for cluster in set(df[cluster_var]):
        clusters.append(cluster)

        df_filter = df[df[cluster_var] == cluster]

        # the matrix is the embedding (it can be any embedding)

        dims = np.arange(0, dim_lenght)
        dims = [str(x) for x in dims]
        matrix = df_filter[dims].values

        # Compute the centroid
        centroid = np.average(matrix, axis=0)

        # Find the index in the matrix of the clostest point
        closest_index = closest_point(centroid, matrix, n=top_elements)

        # Find the index in the matrix of the farest point
        far_index = farest_point(centroid, matrix)

        # find the distance bewteen the centroid and the farest point
        distance = np.linalg.norm(matrix[far_index] - matrix[closest_index])
        distances.append(distance)

        # find the corresponding element to the centroid in the metadata
        centroid_element = df_filter.iloc[closest_index][text_var].tolist()

        # Get rid of empty list
        centroid_element = [x for x in centroid_element if str(x) != "nan"]

        if len(centroid_element) > 1:
            centroid_element = " || ".join(centroid_element)
        else:
            centroid_element = centroid_element[0]
            pass

        centroid_elements.append(centroid_element)

        # find the corresponding element to the farest point in the metadata
        far_element = df_filter.iloc[far_index][text_var]
        far_elements.append(far_element)

    # Create a DataFrame with the elements
    df_centroids = pd.DataFrame(
        {
            "clusters": clusters,
            "centroid_docs": centroid_elements,
            "farest_doc": far_elements,
            "cluster_radius": distances,
        }
    )

    df_centroids = df_centroids.reset_index(drop=True)

    return df_centroids
