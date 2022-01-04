import os, warnings
from joblib import dump, load


def load_data(fpath):
    with open(fpath, 'rb') as in_file:
        return load(in_file)


def save_data(fpath, data):
    with open(fpath, 'wb') as out_file:
        dump(data, out_file)


def _retrieve_input(input_folder, input_type):
    input_path = os.path.join(input_folder, input_type)
    if os.path.exists(input_path):
        return load_data(input_path)
    else:
        raise RuntimeError('{} not found. Are you sure you correctly ran the pre-clustering?')


def retrieve_input(config, folder, dict_key, split_samples, split):
    if dict_key not in config:
        warnings.warn('Input type not detected. The original inputs will be passed to the algo.')
        inputs = split_samples[split]
        input_type = make_split_name('original', split)
    else:
        input_type = config[dict_key]
        inputs = _retrieve_input(folder, make_split_name(input_type, split))
    return inputs, input_type


def make_split_name(base_name, split):
    return base_name + '_{}'.format(split)


def generate_preprocessing_instance(configs, splits):
    for cfg in configs:
        for split in splits.keys():
            yield cfg, split
