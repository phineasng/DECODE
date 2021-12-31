import functools


def preprocess_split(fn):
    @functools.wraps(fn)
    def pre_proc_split(split, model, dataset):
        processed_samples = {}
        for k, samples in split.items():
            processed_samples[k] = fn(samples, model, dataset)
        return processed_samples
    return pre_proc_split


