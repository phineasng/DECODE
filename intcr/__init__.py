"""Module initialization."""

__version__ = "0.1.0"


from intcr.models import MODEL_LOADERS
from intcr.clustering import PRE_CLUSTER_TRANSFORM_REGISTRY, CLUSTERING_ALGOS_REGISTRY, CLUSTERING_EVALUATION_REGISTRY,\
    DATA_VISUALIZATION_REGISTRY, CLUSTERING_ALGOS_CENTERS_GET_FN, CLUSTERING_EVALUATION_MULTIPLIER
from intcr.data import DATASET_GETTERS, CATEGORICAL_ALPHABETS
