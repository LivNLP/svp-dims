import pickle
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, roc_curve

from datavecloader import *
from vec_utils import *


def visualise(vecs_instance, id2label, size=None, output="test"):
    if size is None:
        plt.figure(figsize=(16, 12))
    else:
        plt.figure(figsize=size)

    xticklabels = [
        "" if (axis + 1) % 10 != 0 else str(axis + 1)
        for axis in range(vecs_instance.shape[1])
    ]

    num_true = sum([1 if item[-1] == "1" else 0 for inst_id, item in id2label.items()])

    heatmap = sns.heatmap(
        vecs_instance,
        annot=False,
        fmt=".2f",
        cmap="OrRd",
        xticklabels=xticklabels,
        yticklabels=False,
        cbar=False,
    )

    plt.axhline(y=num_true, color="black", linewidth=8)

    size = 40
    plt.xlabel("Axis", fontsize=size)
    plt.xticks(fontsize=size)
    cbar = plt.colorbar(heatmap.collections[0], ax=heatmap.axes)
    cbar.ax.tick_params(labelsize=size)

    plt.tight_layout()
    plt.show()
    plt.savefig(f"instances_{output}.png")
    plt.clf()
    plt.close()


def convert_axis_into_gold_pred(
    num_positive: int, vec_each_axis
) -> Tuple[np.array, np.array]:
    N = len(vec_each_axis)
    gold = np.zeros(N)
    pred = np.zeros(N)
    for vec_id in range(N):
        if vec_id < num_positive:
            gold[vec_id] = 1
        else:
            gold[vec_id] = 0
    assert (
        np.sum(gold) == num_positive
    ), f"num_positive: {num_positive}, gold: {np.sum(gold)}"

    for curr_num, vec_id in enumerate(np.argsort(vec_each_axis)):
        if curr_num < num_positive:
            pred[vec_id] = 1
        else:
            pred[vec_id] = 0
    assert (
        np.sum(pred) == num_positive
    ), f"num_positive: {num_positive}, gold: {np.sum(pred)}"

    return gold, pred


def evaluate_axis(num_positive: int, vec_each_axis):
    gold, pred = convert_axis_into_gold_pred(num_positive, vec_each_axis)

    return accuracy_score(gold, pred)


def evaluate_all(num_positive: int, vec_mat):
    acc_list = []
    for vec_each_axis in vec_mat:
        acc_list.append(evaluate_axis(num_positive, vec_each_axis))
    return acc_list


def visualise_instance(file_path: str, conduct_ica: bool = True, output: str = 'test'):
    dvl = pickle.load(open(file_path, "rb"))
    vecs = dvl.vecs
    vecs_decomposed, components = decompose_skewness(
        vecs, conduct_ica=conduct_ica, dimension=1024
    )
    vecs_instance_ica = calculate_instance(dvl, vecs_decomposed)
    vecs_instance_raw = calculate_instance(dvl, vecs)

    id2label = {inst_id: item[-1] for inst_id, item in dvl.id2inst.items()}

    n_true = sum([1 if item[-1] == "1" else 0 for inst_id, item in dvl.id2inst.items()])
    n_false = sum(
        [1 if item[-1] == "0" else 0 for inst_id, item in dvl.id2inst.items()]
    )

    vecs_true_ica, id2label_true = get_vecs_by_label(
        vecs_instance_ica, id2label, num_instance=n_true, label=True
    )
    vecs_false_ica, id2label_false = get_vecs_by_label(
        vecs_instance_ica, id2label, num_instance=n_false, label=False
    )
    vecs_true_raw, _ = get_vecs_by_label(
        vecs_instance_raw, id2label, num_instance=n_true, label=True
    )
    vecs_false_raw, _ = get_vecs_by_label(
        vecs_instance_raw, id2label, num_instance=n_false, label=False
    )

    id2label_true_false = {}
    for inst_id, inst_label in id2label_true.items():
        id2label_true_false[inst_id] = inst_label
    for inst_id, inst_label in id2label_false.items():
        id2label_true_false[inst_id] = inst_label

    vecs_ica = np.vstack([vecs_true_ica, vecs_false_ica])
    vecs_raw = np.vstack([vecs_true_raw, vecs_false_raw])

    _ca = "ica" if conduct_ica else "pca"
    visualise(process_each_axis(vecs_ica)[:, :50], id2label_true_false, output=f"{output}_{_ca}")
    if conduct_ica is False:
        visualise(process_each_axis(vecs_ica)[:, -50:], id2label_true_false, output=f"{output}_{_ca}_bottom50")
    visualise(process_each_axis(vecs_raw)[:, :50], id2label_true_false, output=f"{output}_raw")

    accs_ica = evaluate_all(n_true, vecs_ica.T)
    accs_raw = evaluate_all(n_true, vecs_raw.T)

    x_range = range(len(accs_ica))
    plt.scatter(x_range, accs_raw, label="raw (original order)")
    if conduct_ica:
        plt.scatter(x_range, accs_ica, label="ica (sorted by skewness)")
    else:
        plt.scatter(x_range, accs_ica, label="pca")
    plt.xlabel("Dimension id")
    plt.ylabel("Accuracy (WiC)")
    plt.legend()
    plt.show()
    plt.savefig(f"acc_{output}_{_ca}.png")
    plt.clf()
    plt.close()


def sort_vecs_by_true_false(vecs, dvl):
    # id2label = {inst_id: item[-1] for inst_id, item in dvl.id2inst.items()}
    ids_true = [_id for _id in dvl.id2inst.keys() if dvl.id2inst[_id][-1] == "1"]
    ids_false = [_id for _id in dvl.id2inst.keys() if dvl.id2inst[_id][-1] == "0"]

    vecs_instance = calculate_instance(dvl, vecs)
    vecs_true = vecs_instance[ids_true]
    vecs_false = vecs_instance[ids_false]
    return np.vstack([vecs_true, vecs_false])


def convert_axis_into_gold_pred_multiple(
    num_positive: int, vec_multiple_axes
) -> Tuple[np.array, np.array]:
    N = len(vec_multiple_axes)
    vec_each_axis = np.zeros((N))
    for i in range(N):
        vec_each_axis[i] = np.linalg.norm(vec_multiple_axes[i])
    gold, pred = convert_axis_into_gold_pred(num_positive, vec_each_axis)
    return gold, pred


def evaluate_axis_multiple(num_positive: int, vec_multiple_axes):
    gold, pred = convert_axis_into_gold_pred_multiple(num_positive, vec_multiple_axes)

    return accuracy_score(gold, pred)


def evaluate_all_top_x(num_positive: int, vec_mat):
    acc_list = []
    for curr_dim in range(1, len(vec_mat[0]) + 1):
        acc_list.append(evaluate_axis_multiple(num_positive, vec_mat[:, :curr_dim]))
    return acc_list


def evaluate(vecs_raw, vecs_pca, vecs_ica, dvl):
    n_true = sum([1 if item[-1] == "1" else 0 for inst_id, item in dvl.id2inst.items()])
    vecs_true_false_raw = sort_vecs_by_true_false(vecs_raw, dvl)
    vecs_true_false_pca = sort_vecs_by_true_false(vecs_pca, dvl)
    vecs_true_false_ica = sort_vecs_by_true_false(vecs_ica, dvl)
    acc_list_raw = evaluate_all_top_x(n_true, vecs_true_false_raw)
    acc_list_pca = evaluate_all_top_x(n_true, vecs_true_false_pca)
    acc_list_ica = evaluate_all_top_x(n_true, vecs_true_false_ica)
    return acc_list_raw, acc_list_pca, acc_list_ica


def _visualise_acc_multiple(
        acc_list_raw, acc_list_pca, acc_list_ica, ylabel: str = None, output: str = "test"
):
    print(f"raw: {acc_list_raw[-1]}")
    print(f"pca: max = {max(acc_list_pca)}, min = {min(acc_list_pca)}")
    print(f"ica: max = {max(acc_list_ica)}, min = {min(acc_list_ica)}")
    x_range = range(1, len(acc_list_raw) + 1)
    # plt.plot(x_range, [acc_list_raw[-1]] * len(acc_list_raw), label="raw")
    plt.scatter(x_range, acc_list_pca, label="PCA")
    plt.scatter(x_range, acc_list_ica, label="ICA")
    plt.plot(
        x_range,
        [acc_list_raw[-1]] * len(acc_list_raw),
        label="Raw",
        color="black",
        linewidth=3,
    )
    plt.xlabel("Number of axes")
    if ylabel is None:
        plt.ylabel("Accuracy (WiC)")
    else:
        plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    plt.savefig(f"acc_multi_{output}.png")
    plt.clf()
    plt.close()


def visualise_acc_multiple(vecs_raw, vecs_pca, vecs_ica, dvl, output: str = "test"):
    acc_list_raw, acc_list_pca, acc_list_ica = evaluate(
        vecs_raw, vecs_pca, vecs_ica, dvl
    )
    _visualise_acc_multiple(acc_list_raw, acc_list_pca, acc_list_ica, output=output)


def _visualise_roc(
        y_gold, y_pred_raw, y_pred_xcas, is_pca: bool = True, y_pred_ycas=None, output: str = "test"
):
    plt.figure()
    fpr_raw, tpr_raw, _ = roc_curve(y_gold, y_pred_raw)
    roc_auc_raw = auc(fpr_raw, tpr_raw)
    plt.plot(
        fpr_raw, tpr_raw, color="black", lw=4, label=f"Raw (area = {roc_auc_raw:.3f})"
    )

    ratios = ["5%", "10%", "20%", "50%", "100%"]
    label = "PCA" if is_pca else "ICA"
    for i in range(len(y_pred_xcas)):
        fpr_xca, tpr_xca, _ = roc_curve(y_gold, y_pred_xcas[i])
        roc_auc_xca = auc(fpr_xca, tpr_xca)
        print(f"{label}, top-{ratios[i]} (area = {roc_auc_xca:.3f})")
        plt.plot(
            fpr_xca,
            tpr_xca,
            lw=2,
            linestyle="--",
            label=f"{label}, top-{ratios[i]} (area = {roc_auc_xca:.3f})",
        )

    if y_pred_ycas is not None:
        label = "ICA" if is_pca else "PCA"
        for i in range(len(y_pred_ycas)):
            fpr_yca, tpr_yca, _ = roc_curve(y_gold, y_pred_ycas[i])
            roc_auc_yca = auc(fpr_yca, tpr_yca)
            print(f"{label}, top-{ratios[i]} (area = {roc_auc_yca:.3f})")
            plt.plot(
                fpr_yca,
                tpr_yca,
                lw=2,
                linestyle="--",
                label=f"{label}, top-{ratios[i]} (area = {roc_auc_yca:.3f})",
            )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f"roc_{output}.png")
    plt.clf()
    plt.close()


def visualise_roc(path_dvl: str, output: str = "test"):
    dvl = pickle.load(open(path_dvl, "rb"))
    vecs_raw = dvl.vecs
    vecs_pca, _, _ = decompose_skewness(
        dvl.vecs, conduct_ica=False, dimension=1024, return_exp_ratio=True
    )
    vecs_ica, _, _ = decompose_skewness(
        dvl.vecs, conduct_ica=True, dimension=1024, return_exp_ratio=True
    )
    visualise_acc_multiple(vecs_raw, vecs_pca, vecs_ica, dvl, output=output)

    n_true = sum([1 if item[-1] == "1" else 0 for inst_id, item in dvl.id2inst.items()])
    vecs_true_false_raw = sort_vecs_by_true_false(vecs_raw, dvl)
    vecs_true_false_pca = sort_vecs_by_true_false(vecs_pca, dvl)
    vecs_true_false_ica = sort_vecs_by_true_false(vecs_ica, dvl)

    y_gold = [1 if currId < n_true else 0 for currId in range(len(dvl.id2inst.keys()))]
    assert sum(y_gold) == n_true
    y_pred_raw = np.zeros(len(y_gold))
    for i in range(len(y_gold)):
        y_pred_raw[i] = -np.linalg.norm(vecs_true_false_raw[i])

    y_pred_pcas = []
    y_pred_icas = []
    for rate in (0.05, 0.1, 0.2, 0.5, 1.0):
        y_pred_pca = np.zeros(len(y_gold))
        y_pred_ica = np.zeros(len(y_gold))
        dim = int(len(vecs_true_false_pca[0]) * rate)
        print(dim)
        for i in range(len(y_gold)):
            y_pred_pca[i] = -np.linalg.norm(vecs_true_false_pca[i][:dim])
            y_pred_ica[i] = -np.linalg.norm(vecs_true_false_ica[i][:dim])
        y_pred_pcas.append(y_pred_pca)
        y_pred_icas.append(y_pred_ica)

    _visualise_roc(y_gold, y_pred_raw, y_pred_pcas, is_pca=True, output=f"{output}_pca")
    _visualise_roc(y_gold, y_pred_raw, y_pred_icas, is_pca=False, output=f"{output}_ica")
    _visualise_roc(
        y_gold, y_pred_raw, y_pred_pcas, is_pca=True, y_pred_ycas=y_pred_icas, output=f"{output}_all"
    )
