import os
from intcr.pipeline.config import simple_key_check
from intcr.clustering import CLUSTERING_ALGOS_REGISTRY, CLUSTERING_EVALUATION_REGISTRY, \
    CLUSTERING_EVALUATION_MULTIPLIER, DATA_VISUALIZATION_REGISTRY
from intcr.pipeline.utils import load_data, save_data, retrieve_input, generate_preprocessing_instance
import numpy as np
from collections import defaultdict
from ClusterEnsembles import cluster_ensembles
import matplotlib.pyplot as plt
import seaborn as sns


CLUSTERING_METHOD_KEY = 'method'
CLUSTERING_PARAMS_KEY = 'params'
CLUSTERING_INPUT_TYPE_KEY = 'input_type'

CONSENSUS_METHOD_KEY = 'method'
CONSENSUS_PARAMS_KEY = 'params'

SELECTION_METHOD_KEY = 'method'
SELECTION_PARAMS_KEY = 'params'
SELECTION_INPUT_KEY = 'input_type'

VISUALIZATION_METHOD_KEY = 'method'
VISUALIZATION_PARAMS_KEY = 'params'
VISUALIZATION_INPUT_KEY = 'input_type'


def check_cluster_config(config):
    simple_key_check(config, CLUSTERING_METHOD_KEY)
    simple_key_check(config, CLUSTERING_PARAMS_KEY)


def check_consensus_config(config):
    simple_key_check(config, CONSENSUS_METHOD_KEY)


def retrieve_results(path):
    if os.path.exists(path):
        return load_data(path)
    else:
        return dict()


def clustering(clustering_root, preclustering_root, config, split_samples, recompute=False):
    """
    Routine to go through all the transforms to be used for the next clustering step
    """
    if isinstance(config, dict):
        config = [config]

    clustering_assignments = defaultdict(dict)
    clustering_centers = defaultdict(dict)

    def parallelizable_clustering(conf, split):
        check_cluster_config(conf)

        method_name = conf[CLUSTERING_METHOD_KEY]
        params = conf[CLUSTERING_PARAMS_KEY]

        inputs, input_type = retrieve_input(conf, preclustering_root, CLUSTERING_INPUT_TYPE_KEY, split_samples, split)
        clustering_id = '{}_{}'.format(method_name, input_type)
        clustering_id_with_split = '{}_{}'.format(clustering_id, split)
        clusters_assign_path = os.path.join(clustering_root, '{}_assignments'.format(clustering_id_with_split))
        clusters_center_path = os.path.join(clustering_root, '{}_centers'.format(clustering_id_with_split))
        model_path = os.path.join(clustering_root, 'clustermodel_{}'.format(clustering_id_with_split))

        if recompute or \
                not (os.path.exists(clusters_assign_path) and os.path.exists(clusters_center_path) and
                    os.path.exists(model_path)):
            cluster_model = CLUSTERING_ALGOS_REGISTRY[method_name](**params)
            split_clusters = cluster_model.fit_predict(inputs)
            split_centers = cluster_model
            save_data(model_path, cluster_model)
        else:
            cluster_model = load_data(model_path)
            split_clusters = load_data(clusters_assign_path)
            split_centers = load_data(clusters_center_path)
        result = {
            'centers': {clustering_id: {split: split_centers}},
            'labels': {clustering_id: {split: split_clusters}}
        }
        return result

    results = []
    for c, s in generate_preprocessing_instance(config, split_samples.keys()):
        results.append(parallelizable_clustering(c, s))

    for r in results:
        for k in r[0].keys():
            clustering_centers[k].update(r[0][k])
            clustering_assignments[k].update(r[1][k])

    return clustering_centers, clustering_assignments


def consensus_clustering(results_root, assignments, config, recompute=False):
    if isinstance(config, dict):
        config = [config]

    # aggregate per split
    per_split = defaultdict(list)
    per_split_method_idx = defaultdict(dict)
    cluster_methods = list()

    for cluster_method, results in assignments.items():
        idx = 0
        for split, labels in results.items():
            per_split[split].append(labels)
            per_split_method_idx[split][cluster_method] = idx
            idx = idx + 1
        cluster_methods.append(cluster_method)

    cluster_methods = sorted(cluster_methods)
    aggregated = {split: np.stack(labels, axis=0) for split, labels in per_split.items()}
    consensus_assignments = dict()

    if len(cluster_methods) <= 1:
        return consensus_assignments

    # run consensus
    for cfg in config:
        check_consensus_config(cfg)
        params = cfg.get(CONSENSUS_PARAMS_KEY, {})

        consensus_name = cfg[CONSENSUS_METHOD_KEY]
        consensus_assign_path = os.path.join(results_root, '{}_assignments'.format(consensus_name))
        consensus_meta_path = os.path.join(results_root, '{}.meta'.format(consensus_name))

        if not recompute:
            split_clusters = retrieve_results(consensus_assign_path)
            split_meta = sorted(list(retrieve_results(consensus_meta_path)))
        else:
            split_clusters = dict()
            split_meta = list()

        if len(split_clusters) != len(aggregated):
            for split, labels in aggregated.items():
                if recompute or split_meta != cluster_methods or split not in split_clusters:
                    split_clusters[split] = cluster_ensembles(labels, solver=consensus_name, **params)

        consensus_assignments[consensus_name] = split_clusters

        save_data(consensus_assign_path, split_clusters)
        save_data(consensus_meta_path, split_meta)

    return consensus_assignments


def select_best_clustering(assignments, clustering_root, preclustering_root, config: dict,
                           split_samples, recompute=False):
    if len(assignments) == 1:
        # ugly - is there a way to get the key if dict has only one element?
        for cluster_method in assignments.keys():
            return cluster_method

    inputs, input_type = retrieve_input(config, preclustering_root, SELECTION_INPUT_KEY, split_samples)
    scores = dict()

    method = config[SELECTION_METHOD_KEY]
    params = config.get(SELECTION_PARAMS_KEY, {})

    scores_fpath = os.path.join(clustering_root, '{}'.format(method))

    if os.path.exists(scores_fpath) and not recompute:
        scores = load_data(scores_fpath)
        for cluster_method in assignments.keys():
            found = False
            for split, best in scores.items():
                if cluster_method == best[CLUSTERING_METHOD_KEY]:
                    found = True
            if not found:
                recompute = True
                break
    else:
        recompute = True

    if recompute:
        for split, samples in inputs.items():
            methods = []
            curr_scores = []
            for cluster_method, split_labels in assignments.items():
                methods.append(cluster_method)
                curr_scores.append(CLUSTERING_EVALUATION_REGISTRY[method](samples, split_labels[split], **params))
            best_idx = np.argmax(CLUSTERING_EVALUATION_MULTIPLIER[method]*np.array(curr_scores))
            scores[split] = {
                CLUSTERING_METHOD_KEY: methods[best_idx],
                'score': curr_scores[best_idx]
            }
        save_data(scores_fpath, scores)

    return scores


def visualize_clusters(assignments, clustering_root, preclustering_root, config, split_samples, recompute=False):
    if isinstance(config, dict):
        config = [config]

    for cfg in config:

        method = cfg[VISUALIZATION_METHOD_KEY]
        params = cfg.get(VISUALIZATION_PARAMS_KEY, {})
        params['n_components'] = 2
        inputs, input_type = retrieve_input(cfg, preclustering_root, VISUALIZATION_INPUT_KEY, split_samples)

        projection_fn = DATA_VISUALIZATION_REGISTRY[method](**params)

        for split, samples in inputs.items():
            projections = projection_fn.fit_transform(samples)

            for cluster_method, split_labels in assignments.items():

                img_title = os.path.join(clustering_root, '{}_{}_{}'.format(method, split, cluster_method))
                img_path = img_title + '.png'

                if recompute or not os.path.exists(img_path):
                    plt.figure()
                    sns.scatterplot(x=projections[:, 0], y=projections[:, 1], hue=split_labels[split])
                    plt.title(img_title)
                    plt.legend()
                    plt.savefig(img_path, dpi=300, bbox_inches='tight')

