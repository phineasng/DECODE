from intcr import (
    MODEL_LOADERS,
    DATASET_GETTERS,
    CATEGORICAL_ALPHABETS,
    DATA_VISUALIZATION_REGISTRY,
    CLUSTERING_EVALUATION_REGISTRY,
    CLUSTERING_ALGOS_REGISTRY,
    PRE_CLUSTER_TRANSFORM_REGISTRY,
    CLUSTERING_ALGOS_CENTERS_GET_FN,
    CLUSTERING_EVALUATION_MULTIPLIER,
)
import sys
import os
from importlib import machinery, util


def load_module_from_file(fpath):
    loader = machinery.SourceFileLoader('module', fpath)
    spec = util.spec_from_loader('module', loader)
    module = util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def update_model_registry(models_fpath):
    model_module = load_module_from_file(models_fpath)
    MODEL_LOADERS.update(model_module.MODEL_LOADERS)


def update_dataset_registry(dataset_fpath):
    dataset_module = load_module_from_file(dataset_fpath)
    DATASET_GETTERS.update(dataset_module.DATASET_GETTERS)
    CATEGORICAL_ALPHABETS.update(dataset_module.CATEGORICAL_ALPHABETS)


def update_clustering_registry(clustering_fpath):
    clustering_module = load_module_from_file(clustering_fpath)

    PRE_CLUSTER_TRANSFORM_REGISTRY.update(clustering_module.PRECLUSTER_TRANSFORM)
    CLUSTERING_ALGOS_REGISTRY.update(clustering_module.CLUSTERING_ALGOS)
    CLUSTERING_EVALUATION_REGISTRY.update(clustering_module.CLUSTERING_EVALUATORS)
    CLUSTERING_EVALUATION_MULTIPLIER.update(clustering_module.CLUSTERING_EVALUATION_MULTIPLIER)
    CLUSTERING_ALGOS_CENTERS_GET_FN.update(clustering_module.CLUSTERING_ALGOS_CENTERS_GET_FN)


def update_visualization_registry(visualization_fpath):
    visualization_module = load_module_from_file(visualization_fpath)
    DATA_VISUALIZATION_REGISTRY.update(visualization_module.VISUALIZATION_FUNCTIONS)


MODULE_FILES_AND_FN = {
    'model.py': update_model_registry,
    'data.py': update_dataset_registry,
    'clustering.py': update_clustering_registry,
    'visualization.py': update_visualization_registry
}


def update_registries(user_directory=None):
    sys.path.append(user_directory)
    for fname, update_fn in MODULE_FILES_AND_FN.items():
        fpath = os.path.join(user_directory, fname)
        if os.path.exists(fpath):
            update_fn(fpath)
    sys.path.pop()
