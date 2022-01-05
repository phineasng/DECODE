import os
from intcr.pipeline.config import simple_key_check
from intcr.clustering import CLUSTERING_ALGOS_REGISTRY, CLUSTERING_EVALUATION_REGISTRY, \
    CLUSTERING_EVALUATION_MULTIPLIER, DATA_VISUALIZATION_REGISTRY, CLUSTERING_ALGOS_CENTERS_GET_FN
from intcr.pipeline.utils import load_data, save_data, retrieve_input, generate_preprocessing_instance
import numpy as np
from collections import defaultdict
from ClusterEnsembles import cluster_ensembles
import matplotlib.pyplot as plt
import seaborn as sns


CLUSTERING_METHOD_KEY = 'method'
CLUSTERING_PARAMS_KEY = 'params'
CLUSTERING_INPUT_TYPE_KEY = 'input_type'
CLUSTERING_PARAM4ID_KEY = 'param4identification'

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
        param4id = conf.get(CLUSTERING_PARAM4ID_KEY, None)

        inputs, input_type = retrieve_input(conf, preclustering_root, CLUSTERING_INPUT_TYPE_KEY, split_samples, split)
        clustering_id = '{}_{}'.format(method_name, input_type)
        if param4id is not None:
            p4id = params.get(param4id, None)
            if p4id is not None:
                clustering_id = clustering_id + '_{}_{}'.format(param4id, p4id).replace('.', '_')
        clustering_id_with_split = '{}_split{}'.format(clustering_id, split)
        clusters_assign_path = os.path.join(clustering_root, '{}_assignments'.format(clustering_id_with_split))
        clusters_center_path = os.path.join(clustering_root, '{}_centers'.format(clustering_id_with_split))
        model_path = os.path.join(clustering_root, 'clustermodel_{}'.format(clustering_id_with_split))

        if recompute or \
                not (os.path.exists(clusters_assign_path) and os.path.exists(clusters_center_path) and
                    os.path.exists(model_path)):
            cluster_model = CLUSTERING_ALGOS_REGISTRY[method_name](**params)
            split_clusters = cluster_model.fit_predict(inputs)
            split_centers = CLUSTERING_ALGOS_CENTERS_GET_FN[method_name](cluster_model)
            save_data(model_path, cluster_model)
        else:
            split_clusters = load_data(clusters_assign_path)
            split_centers = load_data(clusters_center_path)
        result = {
            'labels': {clustering_id: {split: split_clusters}}
        }
        if split_centers is not None:
            result.update({'centers': {clustering_id: {split: split_centers}}})
        return result

    results = []
    for c, s in generate_preprocessing_instance(config, split_samples.keys()):
        results.append(parallelizable_clustering(c, s))

    for r in results:
        for k in r['labels'].keys():
            if 'centers' in r:
                clustering_centers[k].update(r['centers'][k])
            clustering_assignments[k].update(r['labels'][k])

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

    method = config[SELECTION_METHOD_KEY]
    params = config.get(SELECTION_PARAMS_KEY, {})
    scores_fpath = os.path.join(clustering_root, '{}'.format(method))
    scores = dict()

    if len(assignments) == 1:
        # ugly - is there a way to get the key if dict has only one element?
        for cluster_method in assignments.keys():
            for s in split_samples.keys():
                scores[s] = {
                    CLUSTERING_METHOD_KEY: cluster_method,
                    'score': 0.
                }
        save_data(scores_fpath, scores)
        return scores

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
        def parallelizable_fn(cluster_method, split):
            inputs, input_type = retrieve_input(config, preclustering_root, SELECTION_INPUT_KEY, split_samples, split)
            curr_score = CLUSTERING_EVALUATION_REGISTRY[method](inputs, assignments[cluster_method][split], **params)
            return {
                'method': cluster_method,
                'score': curr_score,
                'split': split
            }

        results = []
        for c, s in generate_preprocessing_instance(assignments.keys(), split_samples.keys()):
            results.append(parallelizable_fn(c, s))

        for s in split_samples.keys():
            curr_methods = []
            curr_scores = []
            for r in results:
                if r['split'] == s:
                    curr_methods.append(r['method'])
                    curr_scores.append(r['score'])
            best_idx = np.argmax(CLUSTERING_EVALUATION_MULTIPLIER[method] * np.array(curr_scores))
            scores[s] = {
                CLUSTERING_METHOD_KEY: curr_methods[best_idx],
                'score': curr_scores[best_idx]
            }
        save_data(scores_fpath, scores)

    return scores


def visualize_clusters(assignments, clustering_root, preclustering_root, config, split_samples, recompute=False):
    if isinstance(config, dict):
        config = [config]

    def parallelizable_fn(cfg, split):
        method = cfg[VISUALIZATION_METHOD_KEY]
        params = cfg.get(VISUALIZATION_PARAMS_KEY, {})
        params['n_components'] = 2
        inputs, input_type = retrieve_input(cfg, preclustering_root, VISUALIZATION_INPUT_KEY, split_samples, split)

        projection_fn = DATA_VISUALIZATION_REGISTRY[method](**params)
        projections = projection_fn.fit_transform(inputs)

        for cluster_method in assignments.keys():

            img_title = '{}_split{}_{}'.format(method, split, cluster_method)
            img_path = os.path.join(clustering_root, img_title + '.png')

            if recompute or not os.path.exists(img_path):
                plt.figure()
                sns.scatterplot(x=projections[:, 0], y=projections[:, 1], hue=assignments[cluster_method][split])
                plt.title(img_title)
                plt.legend()
                plt.savefig(img_path, dpi=300, bbox_inches='tight')

    for c, s in generate_preprocessing_instance(config, split_samples.keys()):
        parallelizable_fn(c, s)
