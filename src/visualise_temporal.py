import pickle

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances

from vec_utils import *
from visualise_contextual import (
    _visualise_acc_multiple,
    _visualise_roc,
    convert_axis_into_gold_pred,
)
from visualise_utils import *
from visualise_utils import _visualise_exp_ratio_all


def load_word2label(path_scd_label):
    word2label = {}
    with open(path_scd_label) as fp:
        for line in fp:
            word, label = line.strip().split("\t")
            word2label[word] = label

    return word2label


def load_word2grade(path_scd_grade):
    word2grade = {}
    with open(path_scd_grade) as fp:
        for line in fp:
            word, grade_str = line.strip().split("\t")
            word2grade[word] = float(grade_str)
    return word2grade


def decompose_scd_vecs(word2vecs_t1, word2vecs_t2):
    vecs_t1 = []
    vecs_t2 = []
    for word in word2vecs_t1.keys():
        vecs_t1.extend(word2vecs_t1[word])
        vecs_t2.extend(word2vecs_t2[word])

    vecs_scd_raw_t1 = np.array(vecs_t1)
    vecs_scd_raw_t2 = np.array(vecs_t2)
    vecs_scd_raw = np.vstack([vecs_scd_raw_t1, vecs_scd_raw_t2])

    vecs_scd_ica, components_scd_ica, exp_ratio_scd_ica = decompose_skewness(
        vecs_scd_raw, conduct_ica=True, dimension=1024, return_exp_ratio=True
    )
    vecs_scd_pca, components_scd_pca, exp_ratio_scd_pca = decompose_skewness(
        vecs_scd_raw, conduct_ica=False, dimension=1024, return_exp_ratio=True
    )

    vecs_scd_ica_t1, vecs_scd_ica_t2 = np.vsplit(vecs_scd_ica, [len(vecs_t1)])
    vecs_scd_pca_t1, vecs_scd_pca_t2 = np.vsplit(vecs_scd_pca, [len(vecs_t1)])

    return (
        vecs_scd_raw_t1,
        vecs_scd_raw_t2,
        vecs_scd_ica_t1,
        vecs_scd_ica_t2,
        components_scd_ica,
        exp_ratio_scd_ica,
        vecs_scd_pca_t1,
        vecs_scd_pca_t2,
        components_scd_pca,
        exp_ratio_scd_pca,
    )


def calculate_instance_scd(
    vecs_t1, vecs_t2, word2vecs_t1, word2vecs_t2, dim: int = None
):
    curr_t1 = 0
    curr_t2 = 0
    vecs_scd_instance = []
    if dim is None:
        dim = len(word2vecs_t1[list(word2vecs_t1.keys())[0]][0])
    for word in word2vecs_t1.keys():
        n_vecs_t1 = len(word2vecs_t1[word])
        n_vecs_t2 = len(word2vecs_t2[word])
        vecs_scd_instance.append(
            np.mean(
                pairwise_distances(
                    vecs_t1[curr_t1 : curr_t1 + n_vecs_t1, :dim],
                    vecs_t2[curr_t2 : curr_t2 + n_vecs_t2, :dim],
                )
            )
        )
        curr_t1 += n_vecs_t1
        curr_t2 += n_vecs_t2
    assert len(vecs_t1) == curr_t1
    assert len(vecs_t2) == curr_t2

    return vecs_scd_instance


def sort_scd_vecs_by_true_false(vecs_scd_instance, word2label):
    ids_true = [
        word_id
        for word_id, word in enumerate(word2label.keys())
        if word2label[word] == "1"
    ]
    ids_false = [
        word_id
        for word_id, word in enumerate(word2label.keys())
        if word2label[word] == "0"
    ]

    vecs_scd_instance = np.array(vecs_scd_instance)
    vecs_true = vecs_scd_instance[ids_true]
    vecs_false = vecs_scd_instance[ids_false]
    return np.concatenate([vecs_true, vecs_false])


def calculate_scd_accuracy(vecs_instance, word2label):
    n_true = sum([1 if word2label[word] == "1" else 0 for word in word2label.keys()])
    vecs_instance = sort_scd_vecs_by_true_false(vecs_instance, word2label)
    scd_gold, scd_pred = convert_axis_into_gold_pred(n_true, -1 * vecs_instance)
    return accuracy_score(scd_gold, scd_pred)


def evaluate_scd_accuracy(
    word2label,
    word2vecs_t1,
    word2vecs_t2,
    vecs_scd_raw_t1,
    vecs_scd_raw_t2,
    vecs_scd_ica_t1,
    vecs_scd_ica_t2,
    vecs_scd_pca_t1,
    vecs_scd_pca_t2,
):
    n_true = sum([1 if word2label[word] == "1" else 0 for word in word2label.keys()])

    acc_list_ica = []
    acc_list_pca = []

    vecs_scd_instance_raw = calculate_instance_scd(
        vecs_scd_raw_t1, vecs_scd_raw_t2, word2vecs_t1, word2vecs_t2, dim=1024
    )
    acc_raw = calculate_scd_accuracy(vecs_scd_instance_raw, word2label)

    for dim in range(1, len(vecs_scd_ica_t1[0]) + 1):
        # for dim in range(1, 50 + 1):
        vecs_scd_instance_ica = calculate_instance_scd(
            vecs_scd_ica_t1, vecs_scd_ica_t2, word2vecs_t1, word2vecs_t2, dim=dim
        )
        vecs_scd_instance_pca = calculate_instance_scd(
            vecs_scd_pca_t1, vecs_scd_pca_t2, word2vecs_t1, word2vecs_t2, dim=dim
        )

        acc_list_ica.append(calculate_scd_accuracy(vecs_scd_instance_ica, word2label))
        acc_list_pca.append(calculate_scd_accuracy(vecs_scd_instance_pca, word2label))

    return acc_raw, acc_list_ica, acc_list_pca


def evaluate_scd_spearman(
    word2grade,
    word2vecs_t1,
    word2vecs_t2,
    vecs_scd_raw_t1,
    vecs_scd_raw_t2,
    vecs_scd_ica_t1,
    vecs_scd_ica_t2,
    vecs_scd_pca_t1,
    vecs_scd_pca_t2,
):

    gold = [word2grade[word] for word in word2vecs_t1.keys()]

    rho_list_ica = []
    rho_list_pca = []

    vecs_scd_instance_raw = calculate_instance_scd(
        vecs_scd_raw_t1, vecs_scd_raw_t2, word2vecs_t1, word2vecs_t2, dim=1024
    )
    rho_raw = spearmanr(vecs_scd_instance_raw, gold)[0]

    for dim in range(1, len(vecs_scd_ica_t1[0]) + 1):
        # for dim in range(1, 50 + 1):
        _vecs_scd_instance_ica = calculate_instance_scd(
            vecs_scd_ica_t1, vecs_scd_ica_t2, word2vecs_t1, word2vecs_t2, dim=dim
        )
        _vecs_scd_instance_pca = calculate_instance_scd(
            vecs_scd_pca_t1, vecs_scd_pca_t2, word2vecs_t1, word2vecs_t2, dim=dim
        )

        rho_ica, _ = spearmanr(_vecs_scd_instance_ica, gold)
        rho_pca, _ = spearmanr(_vecs_scd_instance_pca, gold)
        rho_list_ica.append(abs(rho_ica))
        rho_list_pca.append(abs(rho_pca))

    return rho_raw, rho_list_ica, rho_list_pca


def _evaluate_scd(
    word2label,
    word2grade,
    word2vecs_t1,
    word2vecs_t2,
    vecs_scd_raw_t1,
    vecs_scd_raw_t2,
    vecs_scd_ica_t1,
    vecs_scd_ica_t2,
    vecs_scd_pca_t1,
    vecs_scd_pca_t2,
):
    n_true = sum([1 if word2label[word] == "1" else 0 for word in word2label.keys()])
    gold = [word2grade[word] for word in word2grade.keys()]

    acc_list_ica = []
    acc_list_pca = []

    rho_list_ica = []
    rho_list_pca = []

    vecs_scd_instance_raw = calculate_instance_scd(
        vecs_scd_raw_t1, vecs_scd_raw_t2, word2vecs_t1, word2vecs_t2, dim=1024
    )
    acc_raw = calculate_scd_accuracy(vecs_scd_instance_raw, word2label)
    rho_raw = spearmanr(vecs_scd_instance_raw, gold)[0]

    for dim in range(1, len(vecs_scd_ica_t1[0]) + 1):
        # for dim in range(1, 50 + 1):
        vecs_scd_instance_ica = calculate_instance_scd(
            vecs_scd_ica_t1, vecs_scd_ica_t2, word2vecs_t1, word2vecs_t2, dim=dim
        )
        vecs_scd_instance_pca = calculate_instance_scd(
            vecs_scd_pca_t1, vecs_scd_pca_t2, word2vecs_t1, word2vecs_t2, dim=dim
        )

        acc_list_ica.append(calculate_scd_accuracy(vecs_scd_instance_ica, word2label))
        acc_list_pca.append(calculate_scd_accuracy(vecs_scd_instance_pca, word2label))

        rho_ica, _ = spearmanr(vecs_scd_instance_ica, gold)
        rho_pca, _ = spearmanr(vecs_scd_instance_pca, gold)
        rho_list_ica.append(abs(rho_ica))
        rho_list_pca.append(abs(rho_pca))

    return acc_raw, acc_list_ica, acc_list_pca, rho_raw, rho_list_ica, rho_list_pca


def visualise_scd_roc(path_word2label, path_word2grade, path_scd_t1, path_scd_t2, output = "test"):
    word2label = load_word2label(path_word2label)
    word2grade = load_word2grade(path_word2grade)
    word2vecs_t1 = pickle.load(open(path_scd_t1, "rb"))
    word2vecs_t2 = pickle.load(open(path_scd_t2, "rb"))

    (
        vecs_scd_raw_t1,
        vecs_scd_raw_t2,
        vecs_scd_ica_t1,
        vecs_scd_ica_t2,
        _,
        _,
        vecs_scd_pca_t1,
        vecs_scd_pca_t2,
        _,
        _,
    ) = decompose_scd_vecs(word2vecs_t1, word2vecs_t2)

    n_true = sum([1 if word2label[word] == "1" else 0 for word in word2label.keys()])
    y_gold = [1 if currId < n_true else 0 for currId in range(len(word2label.keys()))]
    assert sum(y_gold) == n_true

    vecs_scd_instance_raw = calculate_instance_scd(
        vecs_scd_raw_t1, vecs_scd_raw_t2, word2vecs_t1, word2vecs_t2, dim=1024
    )
    y_pred_raw = sort_scd_vecs_by_true_false(vecs_scd_instance_raw, word2label)

    y_pred_pcas = []
    y_pred_icas = []
    for rate in (0.05, 0.1, 0.2, 0.5, 1.0):
        dim = int(len(vecs_scd_ica_t1[0]) * rate)
        vecs_scd_instance_ica = calculate_instance_scd(
            vecs_scd_ica_t1, vecs_scd_ica_t2, word2vecs_t1, word2vecs_t2, dim=dim
        )
        vecs_scd_instance_pca = calculate_instance_scd(
            vecs_scd_pca_t1, vecs_scd_pca_t2, word2vecs_t1, word2vecs_t2, dim=dim
        )
        y_pred_pca = sort_scd_vecs_by_true_false(vecs_scd_instance_pca, word2label)
        y_pred_ica = sort_scd_vecs_by_true_false(vecs_scd_instance_ica, word2label),
        y_pred_pcas.append(y_pred_pca)
        y_pred_icas.append(y_pred_ica[0])

    _visualise_roc(y_gold, y_pred_raw, y_pred_pcas, is_pca=True, output=f"{output}_scd_pca")
    _visualise_roc(y_gold, y_pred_raw, y_pred_icas, is_pca=False, output=f"{output}_scd_ica")
    _visualise_roc(
        y_gold, y_pred_raw, y_pred_pcas, is_pca=True, y_pred_ycas=y_pred_icas, output=f"{output}_scd_all",
    )


def evaluate_scd(path_word2label, path_word2grade, path_scd_t1, path_scd_t2, output: str ="test"):
    word2label = load_word2label(path_word2label)
    word2grade = load_word2grade(path_word2grade)
    word2vecs_t1 = pickle.load(open(path_scd_t1, "rb"))
    word2vecs_t2 = pickle.load(open(path_scd_t2, "rb"))

    (
        vecs_scd_raw_t1,
        vecs_scd_raw_t2,
        vecs_scd_ica_t1,
        vecs_scd_ica_t2,
        components_scd_ica,
        exp_ratio_scd_ica,
        vecs_scd_pca_t1,
        vecs_scd_pca_t2,
        components_scd_pca,
        exp_ratio_scd_pca,
    ) = decompose_scd_vecs(word2vecs_t1, word2vecs_t2)

    _visualise_exp_ratio_all(exp_ratio_scd_pca, exp_ratio_scd_ica, output=f"{output}_scd")

    acc_raw, acc_list_ica, acc_list_pca, rho_raw, rho_list_ica, rho_list_pca = (
        _evaluate_scd(
            word2label,
            word2grade,
            word2vecs_t1,
            word2vecs_t2,
            vecs_scd_raw_t1,
            vecs_scd_raw_t2,
            vecs_scd_ica_t1,
            vecs_scd_ica_t2,
            vecs_scd_pca_t1,
            vecs_scd_pca_t2,
        )
    )

    _visualise_acc_multiple(
        [acc_raw] * len(acc_list_pca),
        acc_list_pca,
        acc_list_ica,
        ylabel="Accuracy (SCD)",
        output=f"{output}_scd_acc_full",
    )
    _visualise_acc_multiple(
        [acc_raw] * len(acc_list_pca[:50]),
        acc_list_pca[:50],
        acc_list_ica[:50],
        ylabel="Accuracy (SCD)",
        output=f"{output}_scd_acc_top50",
    )
    _visualise_acc_multiple(
        [rho_raw] * len(rho_list_pca),
        rho_list_pca,
        rho_list_ica,
        ylabel="Spearman rho (SCD)",
        output=f"{output}_scd_rank_full",
    )
    _visualise_acc_multiple(
        [rho_raw] * len(rho_list_pca[:50]),
        rho_list_pca[:50],
        rho_list_ica[:50],
        ylabel="Spearman rho (SCD)",
        output=f"{output}_scd_rank_top50",
    )
