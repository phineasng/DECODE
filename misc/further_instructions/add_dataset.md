# Add your own dataset to the pipeline

To add a dataset it is simply necessary to implement a routine that takes as input a dictionary
containing all the information to create/load dataset, and returns a tuple containing the inputs and the true labels.
That is, the dataset loader should follow:

```python
def your_dataset_loader(dataset_params):
    # load/create dataset
    # example of infos that could be contained in the params
    # - path to where dataset is stored
    # - URL to download the dataset
    # - infos to do some preprocessing, e.g. normalization, one-hot encoding, BLOSUM encoding, etc...
    ...
    return X, y
```

`X` should be a `numpy.array` that has the shape expected by the model to interpret. 
`y` should be a 1D array containing the labels. The user can actually return `None` instead of `y` since most the pipeline does not use this information.
Note however that in this case it would not be possible to run the testing routine with the flag `--test`.

The loader should then be made available to use in ``DECODE`` by adding it to the dataloader registry in `intcr/data/__init__.py`

```python
from intcr.data.your_dataset import your_dataset_loader

DATASET_GETTERS = {
    ...
    'your_dataset_id': your_dataset_loader,
    ...
}
```

Then it can be used by setting the corresponding entry in the config file

```json
    "dataset_id": "your_dataset_id",
    "dataset_params": { # this is the input dictionary to your loader
      ...
    }
```

## Processing the data

The dataset should return the input as expected natively by the model, in a ``numpy.array``. 
However, there are some steps of the ``DECODE`` pipeline that might take a different input format, e.g. clustering or ``Anchors``.
For these steps, it is possible to define data preprocessing steps. 
The user can define a data processing function of the form:

```python
def your_data_processor(samples, *args, **kwargs):
    ...
    return transformed_samples
```

Minimally, the data processing function should take the ``samples`` to transform, which are in the same format as the input returned by the dataset loader.
However, in the pipeline the function will be called also passing the model. Therefore, the user can have access to the model if the data processor is defined as

```python
def your_data_processor(samples, model, *args, **kwargs):
    ...
    return transformed_samples
```

This can be useful if, for example, you want to use a representation of the data from a hidden layer of a neural network model.
``transformed samples`` can also be a distance matrix.

The data processor can then be added to the corresponding registry in ``intcr/clustering/__init__.py``

```python
from where.your.processor.is.defined import your_data_processor

PRE_CLUSTER_TRANSFORM_REGISTRY = {
    ...
    'your_data_processor_id': your_data_processor
    ...
}
```

and used by setting the config file to:

```json
  "clustering": {
    "cluster_preproc_transform": [
      {
        "transform_fn": "your_data_processor_id",
        "output_type": "levenshtein_matrix"
      },
      ...
    ],
    ...
  }
```

The ``output_type`` entry is used to save the transformed 
samples to file and to identify the type needed by other steps of the pipeline.
It can be any name of choice of the user. More details can be found in the  [overview of the pipeline](../../README.md#overview-of-the-pipeline).

# Anchors alphabet

``Anchors`` expects categorical inputs. 
If a dataset does not natively provide categorical inputs a preprocessing function should be [implemented](#processing-the-data) to turn
the original inputs to categorically encoded ones. An example for ``TITAN`` is the function `blosum_emb2categorical` in `intcr/data/tcr_titan.py` that turns BLOSUM embeddings to categorical variables.

To perform the perturbations, ``Anchors`` also needs to have access to the alphabet. The alphabet is a dictionary mapping the categorical values to their "string meaning", e.g. an amino-acid or nucleotide. 

For example:
```python
alphabet = {
    0: 'A',
    1: 'T',
    2: 'C',
    3: 'G',
}
```

This can be usually hardcoded somewhere in your code. You just need to make it accessible 
to ``Anchors`` by adding it to the alphabet registry in `intcr/data/__init__.py`

```python
from where.your.alphabet.is.defined import your_alphabet

CATEGORICAL_ALPHABETS = {
    ...
    'your_alphabet_id': your_alphabet,
    ...
}
```

In our pipeline, ``your_alphabet`` is allowed to be a `callable`, i.e. a function of the form

```python
def your_alphabet(model):
    # extract alphabet from model
    alphabet = ...
    return alphabet
```

This is done for the cases where tha alphabet is dynamically loaded by the model.


To indicate to ``Anchors`` the alphabet to use, in the config file you can set

```json

  "explainer": {
    "anchors": {
      ...
      "categorical_alphabet": "your_alphabet_id"
      ...
    }
  }
```
