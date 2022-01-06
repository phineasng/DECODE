import os
import numpy as np
from alibi.explainers import AnchorTabular
from collections import defaultdict
from intcr.pipeline.utils import retrieve_input, load_data, save_data
from intcr.data import CATEGORICAL_ALPHABETS


ANCHORS_CONSTRUCTOR_PARAMS_KEY = 'constructor_params'
ANCHORS_EXPLANATION_PARAMS_KEY = 'explanation_params'
ANCHORS_INPUT_KEY = 'input_type'
ANCHORS_CATEGORICAL_ALPHABET_KEY = 'categorical_alphabet'


def generate_anchors(assignments, cluster_centers, best_clustering, config, preprocessing_folder, results_root,
                     model, split_samples, recompute=False, random_seed=None):
    # prepare explainer
    constructor_params = config.get(ANCHORS_CONSTRUCTOR_PARAMS_KEY, {})
    explanation_params = config.get(ANCHORS_EXPLANATION_PARAMS_KEY, {})
    ### - find better solution for this
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
            if best_clustering[split]['method'] not in cluster_centers:
                curr_centers = []
                labels = assignments[best_clustering[split]['method']][split]
                idx = np.arange(len(labels))
                for i in np.unique(labels):
                    curr_centers.append(np.random.choice(idx[labels == i]))
                centers[split] = np.array(curr_centers)
            else:
                centers = cluster_centers[best_clustering[split]['method']][split]

        def generate_split_center_args():
            for sp, cntrs in centers.items():
                for cntr in cntrs:
                    yield sp, cntr

        def parallelizable_fn(spl, cnt):
            inputs, _ = retrieve_input(config, preprocessing_folder, ANCHORS_INPUT_KEY, split_samples, spl)
            partial_anchors_fpath = os.path.join(results_root, 'anchors_split{}_centroid{}.pkl'.format(spl, cnt))
            anchor = anchors_explainer.explain(inputs[cnt], **explanation_params)
            result = {spl: {cnt: anchor}}
            save_data(partial_anchors_fpath, anchor)
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
