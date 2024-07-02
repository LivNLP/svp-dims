from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.stats import skew, zscore
from sklearn.decomposition import PCA, FastICA


def sort_axis_skewness(
    WV: List[List[float]],
    components: List[List[float]],
    dimension: int = None,
    return_skew: bool = False,
):
    """sort axis by skewness (for ICA)
    :param WV: np.array, [V*T, dim] dim
    :param components: np.array, [dim-decomposed, dim] dim
    :param dimension: int, dimension of decomposed
    :param return_skew: bool, return skewness or not
    :return:
    """
    WV_transformed = np.dot(WV, components.T)

    skewness = skew(WV_transformed)
    skewness = np.abs(skewness)
    skewness /= np.sum(skewness)

    if dimension is None:
        sorted_axis_ids = np.argsort(skewness * -1)
    else:
        sorted_axis_ids = np.argsort(skewness * -1)[:dimension]

    if return_skew:
        return components[sorted_axis_ids], skewness[sorted_axis_ids]
    else:
        return components[sorted_axis_ids]


def decompose_skewness(
    WV: List[List[float]],
    conduct_ica: bool = False,
    dimension: int = 3,
    return_exp_ratio: bool = False,
    max_iter: int = 200,
) -> List[List[float]]:
    """decompose
    :param WV: numpy.array, [V, dim] dim
    :param conduct_ica: bool, conduct ICA instead of PCA
    :param dimension: int, dimension of decomposed
    :param return_exp_ratio: bool, return explained ratio / skewness or not
    :return: WV_d: numpy.array, [V, dimension] dim
    """
    if conduct_ica:
        D = len(WV[0])
        # ica = FastICA(n_components=D, random_state=12345)
        ica = FastICA(n_components=D, random_state=12345, max_iter=max_iter)
        WV_d = ica.fit_transform(WV)
        components = ica.components_
        if return_exp_ratio:
            sorted_components, exp_ratio = sort_axis_skewness(
                WV, components, dimension, return_skew=True
            )
        else:
            sorted_components = sort_axis_skewness(WV, components, dimension)
        components = sorted_components
        WV_d = np.dot(WV, sorted_components.T)
    else:
        pca = PCA(n_components=dimension)
        WV_d = pca.fit_transform(WV)
        components = pca.components_
        if return_exp_ratio:
            exp_ratio = pca.explained_variance_ratio_

    if return_exp_ratio:
        return WV_d, components, exp_ratio
    else:
        return WV_d, components


def process_each_axis(w2v):
    max_axis = np.max(w2v, axis=0)
    min_axis = np.min(w2v, axis=0)
    return (w2v - min_axis) / (max_axis - min_axis)


def standardize_each_axis(w2v):
    return zscore(w2v)


def get_vecs_by_label(vecs_instance, id2label, num_instance=50, label=True):
    vecs_label = np.zeros([num_instance, vecs_instance.shape[1]])
    id2label_filtered = {}
    target_label = "1" if label else "0"
    curr_id = 0
    for inst_id in id2label.keys():
        label = id2label[inst_id]
        if label == target_label:
            vecs_label[curr_id] = vecs_instance[inst_id]
            id2label_filtered[inst_id] = label
            curr_id += 1
            if curr_id == num_instance:
                break
    return vecs_label, id2label_filtered
