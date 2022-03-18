# Add a clustering method

A clustering method should behave similarly to the methods implemented in `scikit`. 
In particular, the method should provide the function

```python
class YourClusteringModel:
    def __init__(self, p1, p2, ...):
    ...
    def fit_predict(self, X):
        return labels
    ...
```

that takes the input samples to cluster (possibly [transformed](./add_dataset.md#processing-the-data)) and return the cluster ids for each sample.

Optionally, the model should provide a way to get the ids of the cluster centroids as a 1D `numpy.array`: this could be either an attribute or a getter function (or a more creative way).

To make the clustering model available to ``DECODE``, add it to the cluster algorithms registry in `intcr/clustering/__init__.py`.
If you provide a way to get cluster centers, you should add the method in the corresponding registry in the same file.

```python
from where.your.method.is.defined import YourClusterModel

CLUSTERING_ALGOS_REGISTRY = {
    ...
    'clustering_model_id': YourClusterModel
    ...
}


CLUSTERING_ALGOS_CENTERS_GET_FN = {
    ...
    # use the same id as above
    'clustering_model_id': lambda model: model.get_cluster_centers()
    ...
}
```

Note that if a method to get cluster centers is not provided, a fallback method (e.g. cluster median) should be provided. More details are given [here](./clustering.md).

To use the method in your experiment, you should set the following entries

```json
  ...
  "clustering": {
    ...
    "algos": [
      {
        "method": "clustering_model_id",
              ...
	    "centroid_fallback_selection_method": "median", # optional
        "params": { # parameters to construct the clustering model
          "p1": ...,
          "p2": ...
          ...
        }
      }
    ]
    ...
  }
  ...
```


## Add a clustering scoring method

It is possible to define your own way to score a cluster method. This can be achieved by defining a routine

```python
def your_scoring_fn(samples, labels, ...other parameters):
    # score
    return score
```

Similarly as above, you can add this function in the registry in ``intcr/clustering/__init__.py``.
Note that a multiplier for the score should be provided in the corresponding registry.
The multiplier tells ``DECODE`` how the score should be interpreted, i.e. high score is best (`multipler=1`) or worst (`multiplier=-1`).

```python
from where.your.method.is.defined import YourClusterModel

CLUSTERING_EVALUATION_REGISTRY = {
  ...,
  'scoring_fn_id': your_scoring_fn
  ...
}

CLUSTERING_EVALUATION_MULTIPLIER = {
    ...
    'scoring_fn_id': 1,
    ...
}
```

To use the method in the pipeline, you can set the corresponding entries:

```json
"clustering": {
    ...
    "cluster_selection": {
      "method": "scoring_fn_id",
      "params": {
        other params ...
      }
    }
    ...
  }
```
