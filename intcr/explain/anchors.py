import os
import numpy as np
from alibi.explainers import AnchorTabular
from intcr.pipeline.utils import retrieve_input, load_data, save_data


ANCHORS_CONSTRUCTOR_PARAMS_KEY = 'constructor_params'
ANCHORS_EXPLANATION_PARAMS_KEY = 'explanation_params'
ANCHORS_INPUT_KEY = 'input_type'


def generate_anchors(assignments, cluster_centers, best_clustering, config, preprocessing_folder, results_root,
                     model, dataset, split_samples, recompute=False, random_seed=None):
    constructor_params = config.get(ANCHORS_CONSTRUCTOR_PARAMS_KEY, {})
    explanation_params = config.get(ANCHORS_EXPLANATION_PARAMS_KEY, {})
    n_features = len(dataset[0])
    feature_names = ['{}'.format(i) for i in range(n_features)]

    inputs, _ = retrieve_input(config, preprocessing_folder, ANCHORS_INPUT_KEY, split_samples)
    anchors = dict()
    anchors_explainer = AnchorTabular(
        predictor=model.predict,
        feature_names=feature_names,
        **constructor_params
    ).fit(dataset[:])

    anchors_fpath = os.path.join(results_root, 'anchors.pkl')

    if not os.path.exists(anchors_fpath) or recompute:
        np.random.seed(random_seed)
        for split, labels in assignments[best_clustering].items():
            if best_clustering not in cluster_centers:
                idx = np.arange(len(assignments))
                centers = []
                for i in np.unique(labels):
                    centers.append(np.random.choice(idx[labels == i]))
                centers = np.array(centers)
            else:
                centers = cluster_centers[best_clustering][split]

            samples = inputs[split]
            explanations = []
            for center in centers:
                explanations.append(anchors_explainer.explain(samples[center], **explanation_params))
            anchors[split] = explanations
        save_data(anchors_fpath, anchors)
    else:
        anchors = load_data(anchors_fpath)

    return anchors
