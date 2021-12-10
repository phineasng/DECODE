import os
import warnings
import pickle

# constants
# - step 0. Data Preparation
PREPARATION_SUBDIR = 'data_preparation'
SPLIT_FNAME = 'data_split.pkl'


def setup_folder(root_path):
    if os.path.exists(root_path):
        warnings.warn('{} exists!'.format(root_path))
    os.makedirs(root_path, exist_ok=True)


def setup_preparation_root(root):
    path = os.path.join(root, PREPARATION_SUBDIR)
    setup_folder(path)
    split_path = os.path.join(path, SPLIT_FNAME)
    return path, os.path.exists(split_path)


def retrieve_prepared_data(preparation_root):
    split_path = os.path.join(preparation_root, SPLIT_FNAME)
    with open(split_path, 'rb') as split_file:
        return pickle.load(split_file)


def save_prepared_data(preparation_root, split_data):
    split_path = os.path.join(preparation_root, SPLIT_FNAME)
    with open(split_path, 'wb') as split_file:
        return pickle.dump(split_data, split_file)





