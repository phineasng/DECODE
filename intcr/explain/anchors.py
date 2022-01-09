import os
import numpy as np
import pandas as pd
from alibi.explainers import AnchorTabular
from collections import defaultdict
from intcr.pipeline.utils import retrieve_input, load_data, save_data
from intcr.data import CATEGORICAL_ALPHABETS
import seaborn as sns
import matplotlib.pyplot as plt


ANCHORS_CONSTRUCTOR_PARAMS_KEY = 'constructor_params'
ANCHORS_EXPLANATION_PARAMS_KEY = 'explanation_params'
ANCHORS_INPUT_KEY = 'input_type'
ANCHORS_CATEGORICAL_ALPHABET_KEY = 'categorical_alphabet'


def generate_anchors(assignments, cluster_centers, best_clustering, config, preprocessing_folder, results_root,
                     model, split_samples, recompute=False, random_seed=None):
    # prepare explainer
    constructor_params = config.get(ANCHORS_CONSTRUCTOR_PARAMS_KEY, {})
    explanation_params = config.get(ANCHORS_EXPLANATION_PARAMS_KEY, {})

    # - find better solution for this
    dataset = []
    for s in split_samples.keys():
        inputs, _ = retrieve_input(config, preprocessing_folder, ANCHORS_INPUT_KEY, split_samples, s)
        dataset.append(inputs)
    dataset = np.concatenate(dataset, axis=0)
    n_features = len(dataset[0])
    feature_names = ['{}'.format(i) for i in range(n_features)]
    categorical_names = {
        i: CATEGORICAL_ALPHABETS[config[ANCHORS_CATEGORICAL_ALPHABET_KEY]] for i in range(n_features)
    }

    anchors_explainer = AnchorTabular(
        predictor=model.predict,
        feature_names=feature_names,
        categorical_names=categorical_names,
        **constructor_params
    ).fit(dataset)

    anchors = defaultdict(dict)
    anchors_fpath = os.path.join(results_root, 'anchors.pkl')

    if not os.path.exists(anchors_fpath) or recompute:
        np.random.seed(random_seed)
        # prepare centers
        centers = dict()
        for split in split_samples.keys():
            if best_clustering[split]['method'] not in cluster_centers: # method did not provide centroids
                curr_centers = []
                labels = assignments[best_clustering[split]['method']][split]
                idx = np.arange(len(labels))
                for i in np.unique(labels):
                    curr_centers.append(np.random.choice(idx[labels == i]))
                centers[split] = np.array(curr_centers)
            else:
                centers[split] = cluster_centers[best_clustering[split]['method']][split]

        def generate_split_center_args():
            for sp, cntrs in centers.items():
                for cntr in cntrs:
                    yield sp, cntr

        def parallelizable_fn(spl, cnt):
            partial_anchors_fpath = os.path.join(results_root, 'anchors_split{}_centroid{}.pkl'.format(spl, cnt))
            if os.path.exists(partial_anchors_fpath):
                anchor = load_data(partial_anchors_fpath)
            else:
                inputs, _ = retrieve_input(config, preprocessing_folder, ANCHORS_INPUT_KEY, split_samples, spl)
                anchor = anchors_explainer.explain(inputs[cnt], **explanation_params)
                save_data(partial_anchors_fpath, anchor)
            result = {spl: {cnt: anchor}}
            return result

        results = []
        for s, c in generate_split_center_args():
            results.append(parallelizable_fn(s, c))

        for r in results:
            for s, c_exp in r.items():
                anchors[s].update(c_exp)
        save_data(anchors_fpath, anchors)
    else:
        anchors = load_data(anchors_fpath)

    return anchors


def _feature_set(anchor):
    feature_set = set()
    raw = anchor.data["raw"]
    for feat in raw["feature"]:
        feature_set.add((feat, int(raw["instance"][feat])))
    return feature_set


def _n_feats_overlapping(anchor_1, anchor_2):
    features_1 = _feature_set(anchor_1)
    features_2 = _feature_set(anchor_2)
    return len(features_1.intersection(features_2))


def anchor_verification(anchor, samples):
    """
    Given an anchor, returns which samples fulfill the anchor
    """
    n_samples = len(samples)
    results = np.ones(n_samples, dtype=np.bool)
    features = anchor.data["raw"]["feature"]
    instance = anchor.data["raw"]["instance"]
    for feat in features:
        results = results*(samples[:, feat] == instance[feat])
    return results


def multi_anchor_verification(anchors, samples):
    """
    Given an anchor, returns which samples fulfill the anchor
    """
    n_samples = len(samples)
    results = np.zeros(n_samples, dtype=np.bool)
    for anchor in anchors:
        results += anchor_verification(anchor, samples)
    return results


def evaluate_anchors(anchors, assignments, best_clustering, split_samples, root, preproc_root, config):
    # boiler plate code for submission purposes
    evaluation_df = pd.DataFrame()

    # flattening the explanations nested dictionary, and other infos
    explanation_list = []
    explanation_id_list = [] # explanation identifiers, i.e. (split, idx_in_the_split)
    explanations_per_split = defaultdict(list)
    samples_list_per_cluster = [] # each element is the subset of samples corresponding to a single cluster
    cluster_lengths = []
    split_ids = defaultdict(list)
    n_samples = 0
    samples_list_per_split = {}
    for split, explanations in anchors.items():
        inputs, _ = retrieve_input(config, preproc_root, ANCHORS_INPUT_KEY, split_samples, split)
        labels = assignments[best_clustering[split]['method']][split]
        samples_list_per_split[split] = inputs
        for center_id, expl in explanations.items():
            explanation_list.append(expl)
            explanation_id_list.append((split, center_id))
            split_ids[split].append(center_id)
            center_label = labels[center_id]
            samples_list_per_cluster.append(inputs[labels == center_label])
            cluster_lengths.append(len(samples_list_per_cluster[-1]))
            n_samples += cluster_lengths[-1]
            explanations_per_split[split].append(expl)
    n_explanations = len(explanation_list)
    n_splits = len(anchors)

    overlap_matrix = np.zeros((n_explanations, n_explanations))  # count the number of overlapping rules
    prediction_matrix = np.zeros((n_explanations, n_explanations))  # compute how many samples of a cluster fulfill a certain anchor
    for i in range(n_explanations):
        for j in range(n_explanations):
            if i < j:
                overlap_matrix[i,j] = _n_feats_overlapping(explanation_list[i], explanation_list[j])
            prediction_matrix[i, j] = np.sum(anchor_verification(explanation_list[i], samples_list_per_cluster[j]))
    split_prediction_matrix = np.zeros((n_splits, n_splits))
    for i, sp_i in enumerate(sorted(anchors.keys())):
        for j, sp_j in enumerate(sorted(anchors.keys())):
            split_prediction_matrix[i, j] = np.sum(multi_anchor_verification(explanations_per_split[sp_i], samples_list_per_split[sp_j]))

    cluster_accuracy = dict()
    cluster_precision = dict()
    cluster_recall = dict()

    cluster_split_accuracy = dict()
    cluster_split_precision = dict()
    cluster_split_recall = dict()

    for i in range(n_explanations):
        # evaluate anchor
        anchor_id = explanation_id_list[i]
        cluster_size = cluster_lengths[i]

        tp = prediction_matrix[i,i]
        fp = np.sum(prediction_matrix[i]) - tp
        fn = cluster_size - tp
        tn = (n_samples - cluster_size) - fn

        cluster_accuracy[anchor_id] = (tp+tn)/(tp+tn+fp+fn)
        cluster_precision[anchor_id] = tp/(tp+fp)
        cluster_recall[anchor_id] = tp/(tp+fn)

        cluster_split_tp = 0
        cluster_split_fp = 0
        cluster_split_tn = 0
        cluster_split_fn = 0

        sp_i = anchor_id[0]
        split_size = len(split_samples[sp_i])
        for j in range(n_explanations):
            sp_j = explanation_id_list[j][0]
            if sp_i == sp_j:
                cluster_split_tp += prediction_matrix[i, j]
            cluster_split_fp = np.sum(prediction_matrix[i]) - cluster_split_tp
            cluster_split_fn = split_size - cluster_split_fp
            cluster_split_tn = (n_samples - split_size) - cluster_split_fn

        cluster_split_accuracy[anchor_id] = (cluster_split_tp + cluster_split_tn) / (cluster_split_tp + cluster_split_tn + cluster_split_fp + cluster_split_fn)
        cluster_split_precision[anchor_id] = cluster_split_tp / (cluster_split_tp + cluster_split_fp)
        cluster_split_recall[anchor_id] = cluster_split_tp / (cluster_split_tp + cluster_split_fn)

    split_accuracy = dict()
    split_precision = dict()
    split_recall = dict()
    for i, sp_i in enumerate(sorted(anchors.keys())):
        split_size = len(split_samples[sp_i])

        tp = split_prediction_matrix[i,i]
        fp = np.sum(split_prediction_matrix[i]) - tp
        fn = split_size - tp
        tn = (n_samples - split_size) - fn

        split_accuracy[sp_i] = (tp + tn) / (tp + tn + fp + fn)
        split_precision[sp_i] = tp / (tp + fp)
        split_recall[sp_i] = tp / (tp + fn)

    metrics_ids = ["id", "metric", "value"]
    cluster_metrics = []
    cluster_split_metrics = []
    for i in range(n_explanations):
        anchor_id = explanation_id_list[i]
        cluster_metrics.append([anchor_id, "precision", cluster_precision[anchor_id]])
        cluster_metrics.append([anchor_id, "recall", cluster_recall[anchor_id]])
        cluster_metrics.append([anchor_id, "accuracy", cluster_accuracy[anchor_id]])

        cluster_split_metrics.append([anchor_id, "precision", cluster_split_precision[anchor_id]])
        cluster_split_metrics.append([anchor_id, "recall", cluster_split_recall[anchor_id]])
        cluster_split_metrics.append([anchor_id, "accuracy", cluster_split_accuracy[anchor_id]])
    cluster_metrics_df = pd.DataFrame(np.stack(cluster_metrics), columns=metrics_ids)
    cluster_split_metrics_df = pd.DataFrame(np.stack(cluster_split_metrics), columns=metrics_ids)

    split_metrics = []
    for i, sp_i in enumerate(sorted(split_samples.keys())):
        split_metrics.append([sp_i, "precision", split_precision[sp_i]])
        split_metrics.append([sp_i, "accuracy", split_accuracy[sp_i]])
        split_metrics.append([sp_i, "recall", split_recall[sp_i]])
    split_metrics_df = pd.DataFrame(np.stack(split_metrics), columns=metrics_ids)
    results = {
        "cluster_metrics": cluster_metrics_df,
        "cluster_split_metrics": cluster_split_metrics_df,
        "split_metrics": split_metrics_df
    }
    metrics_fpath = os.path.join(root, 'metrics')
    save_data(metrics_fpath, results)

    # visualize
    def generate_figure(df, figname):
        plt.figure()
        sns.barplot(data=df, x="metric", y="value", hue="id")
        plt.savefig(figname, dpi=300)

    generate_figure(cluster_metrics_df, os.path.join(root, 'cluster_metrics.png'))
    generate_figure(cluster_split_metrics_df, os.path.join(root, 'cluster_split_metrics.png'))
    generate_figure(split_metrics_df, os.path.join(root, 'split_metrics.png'))


    return results
