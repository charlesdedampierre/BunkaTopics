import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans


def compute_knee(data, max_k):
    # Compute sum of squared distances for different values of K
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distortions.append(
            kmeans.inertia_
        )  # Inertia: Sum of squared distances to the nearest centroid

    # Find the optimal number of clusters using the KneeLocator
    knee = KneeLocator(
        range(1, max_k + 1), distortions, curve="convex", direction="decreasing"
    )

    # Plot the elbow curve with the knee point marked
    plt.plot(range(1, max_k + 1), distortions, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Sum of squared distances")
    plt.title("Elbow Curve")
    plt.vlines(
        knee.knee,
        plt.ylim()[0],
        plt.ylim()[1],
        linestyles="dashed",
        colors="r",
        label="Knee",
    )
    plt.legend()

    # Print the optimal number of clusters
    print(f"Optimal number of clusters (K): {knee.knee}")
    plt.show()
