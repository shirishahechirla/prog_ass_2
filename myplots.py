import matplotlib.pyplot as plt
import utils as u

#----------------------------------------------------------------------
def plot_part1C(kmeans_dct, file_nm):
    """
    Deterministic
    Plot the datasets in kmeans_dct . 
    Each row: value of k
    Each column: value a dataset

    Aguments: 
    kmeans_dct: kmeans_dct.keys(): (
    """
    nb_ks = 4
    nb_datasets = len(kmeans_dct.keys())
    print(f"{kmeans_dct=}")
    print(f"{nb_datasets=}")
    print(f"{nb_ks=}")

    figure, axes = plt.subplots(nb_ks, nb_datasets, figsize=(10, 10))

    data_dict = {}

    for i, (key, v) in enumerate(kmeans_dct.items()):
        X, y = v[0]
        y_kmeans = v[1]
        for j, (k, y_k) in enumerate(y_kmeans.items()):
            ax = axes[j, i]
            ax.scatter(X[:, 0], X[:, 1], c=y_k, s=2, cmap="viridis")
            ax.set_title(f"{key}, k={k}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

    plt.tight_layout()
    # plt.show()
    plt.savefig(file_nm)

    return 
