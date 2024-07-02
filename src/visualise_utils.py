import pickle

import matplotlib.pyplot as plt

from vec_utils import *


def visualise_exp_ratio(file_path: str, is_ica: bool = False, output: str = "test"):
    dvl = pickle.load(open(file_path, "rb"))
    vecs = dvl.vecs
    vecs_decomposed, components, exp_ratio = decompose_skewness(
        vecs, conduct_ica=is_ica, dimension=1024, return_exp_ratio=True
    )

    if is_ica:
        plt.ylabel("Explained ratio (skewness)")
    else:
        plt.ylabel("Explained variance ratio")
    plt.xlabel("Dimension id")
    plt.xscale("log")
    plt.scatter(range(len(exp_ratio)), exp_ratio)
    plt.show()
    plt.savefig(f"exp_{output}.png")
    plt.clf()
    plt.close()

    D = 50
    if is_ica:
        plt.ylabel("Explained ratio (skewness)")
    else:
        plt.ylabel("Explained variance ratio")
    plt.xlabel("Dimension id")
    plt.scatter(range(D), exp_ratio[:D])
    plt.show()
    plt.savefig(f"exp_top50_{output}.png")
    plt.clf()
    plt.close()


def _visualise_exp_ratio_all(exp_ratio_pca, exp_ratio_ica, output: str = "test"):

    plt.ylabel("Explained variance/Skewness ratio")
    plt.xlabel("Dimension id")
    plt.xscale("log")
    plt.scatter(range(len(exp_ratio_pca)), exp_ratio_pca, label="PCA")
    plt.scatter(range(len(exp_ratio_ica)), exp_ratio_ica, label="ICA")
    plt.legend()
    plt.show()
    plt.savefig(f"exp_{output}.png")
    plt.clf()
    plt.close()

    D = 50
    plt.ylabel("Explained variance/Skewness ratio")
    plt.xlabel("Dimension id")
    plt.scatter(range(D), exp_ratio_pca[:D], label="PCA")
    plt.scatter(range(D), exp_ratio_ica[:D], label="ICA")
    plt.legend()
    plt.show()
    plt.savefig(f"exp_top50_{output}.png")
    plt.clf()
    plt.close()


def visualise_exp_ratio_all(file_path: str, output: str = "test"):
    dvl = pickle.load(open(file_path, "rb"))
    vecs = dvl.vecs
    vecs_decomposed_pca, components_pca, exp_ratio_pca = decompose_skewness(
        vecs, conduct_ica=False, dimension=1024, return_exp_ratio=True
    )
    vecs_decomposed_ica, components_ica, exp_ratio_ica = decompose_skewness(
        vecs, conduct_ica=True, dimension=1024, return_exp_ratio=True
    )

    _visualise_exp_ratio_all(exp_ratio_pca, exp_ratio_ica, output=output)
