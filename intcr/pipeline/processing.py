"""
Data processing utils for the interpretability pipeline
"""
import torch
import numpy as np
from intcr.models.utils import batch_predict


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
    predictions = batch_predict(dataset, model, batch_size)
    non_bind_idx = (predictions == 0)
    bind_idx = (predictions == 1)
    return {
        0: dataset[non_bind_idx],
        1: dataset[bind_idx]
    }
