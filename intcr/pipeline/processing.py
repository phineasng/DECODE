"""
Data processing utils for the interpretability pipeline
"""
import torch
import numpy as np


def split_bind_nonbind(dataset: np.array, model, batch_size: int=None):
    """
    Split dataset into a binding set and a non-binding set.
    Assuming that binding has label 1, and non-binding has label 0

    Args:
        dataset: numpy array
        model: model used for prediction. Assuming that it has a predict function that returns a numpy array
        batch_size: int denoting if the dataset should be processed in batches. If None, the dataset will be processed
                    in a single step
    """
    if batch_size is None:
        batch_size = len(dataset)

    progress = range(0, len(dataset), batch_size)

    predictions = []
    for i in progress:
        x = dataset[i:i+batch_size]
        predictions.append(model.predict(x))

    predictions = np.concatenate(predictions, axis=0)
    non_bind_idx = (predictions == 0)
    bind_idx = (predictions == 1)
    return {
        0: predictions[non_bind_idx],
        1: predictions[bind_idx]
    }


def preprocess4clustering(split_samples, preprocess_fn, preprocess_params):
    pass
