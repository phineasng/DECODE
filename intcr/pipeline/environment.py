import os
import warnings
from intcr.pipeline.utils import load_data, save_data

# constants
# - step 0. Data Preparation
PREPARATION_SUBDIR = 'data_preparation'
SPLIT_FNAME = 'data_split.pkl'
# - step 1. Clustering
CLUSTERING_SUBDIR = 'clustering'
PREPROCESSING_SUBDIR = 'clustering_preprocessing'
CLUSTERING_RESULTS_SUBDIR = 'clustering_results'

EXPLAINER_SUBDIR = 'explanations'


def setup_folder(root_path):
    if os.path.exists(root_path):
        warnings.warn('{} exists!'.format(root_path))
    os.makedirs(root_path, exist_ok=True)


# Preparation step (0)
def setup_preparation_root(root):
    path = os.path.join(root, PREPARATION_SUBDIR)
    setup_folder(path)
    split_path = os.path.join(path, SPLIT_FNAME)
    return path, os.path.exists(split_path)


def retrieve_prepared_data(preparation_root):
    split_path = os.path.join(preparation_root, SPLIT_FNAME)
    return load_data(split_path)


def save_prepared_data(preparation_root, split_data):
    split_path = os.path.join(preparation_root, SPLIT_FNAME)
    save_data(split_path, split_data)


# Clustering step (1)
def setup_clustering_root(root):
    path = os.path.join(root, CLUSTERING_SUBDIR)
    setup_folder(path)

    preproc_subdir = os.path.join(path, PREPROCESSING_SUBDIR)
    setup_folder(preproc_subdir)

    results_subdir = os.path.join(path, CLUSTERING_RESULTS_SUBDIR)
    setup_folder(results_subdir)

    return preproc_subdir, results_subdir


def setup_explainer_root(root):
    path = os.path.join(root, EXPLAINER_SUBDIR)
    setup_folder(path)
    return path







