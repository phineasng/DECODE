# Add your own model to the pipeline

To include your own model, the most straightforward ways is to implement 2 components:

1. The model itself (possibly unnecessary) - [instructions](#implement-the-model)
2. A model loader - [instructions](#implement-the-model-loader)

The model and model loader should be defined in the python file called `model.py` in a folder of your choice. 
This folder (which we will call `YOUR_FOLDER/` for the remainder of the instructions) should contain all your (optional) customization files, e.g. the files `clustering.py` (as defined [here](./add_clustering_method.md)), `data.py` (as defined [here](./add_dataset.md)), etc..
You can find a template of the folder structure and files that you can customize in `decode/example/template/`.

We will use the `TITAN` model as a working example. 
You can find this implementation in `decode/example/titan/`.

## Implement the model 

For the pipeline to work, we need a model that exposes a `predict(self, x)`
 function, that takes a *batched* input `x` (a `numpy.array`) and returns predicted *labels* (also `numpy.array` but 1D). 
If your model already provides this method (e.g. a `scikit` model), you can skip this step and directly [implement the model loader](#implement-the-model-loader).

<mark>NOTE: The version of `Anchors` in our pipeline expects a 2D *categorical* array (`x.shape = [BATCH, N_FEATS]`) as an input. 
If your model does not natively work with categorical inpu, then you need to wrap it. Please refer to the rest of this section for an example.</mark>

Your model could, however, be more complex, e.g.:
- your model does not natively provide a `predict` function (e.g. `torch` modules)
- your input needs some further preprocessing of the input
- your model natively expects a non-categorical input, or an input that has more than two dimensions

In these cases, we suggest to wrap your model, as we did for `TITAN`. 

``TITAN`` does not natively provide a predict function. Furthermore, the original model expects two inputs: the TCR and the epitope. 
In our example (included in the paper), we are fixing the epitope. 
For our use case, we "fix" this problem by passing the epitope at construction and keep it in the model as an attribute.
The predict function will just then take only the TCR as an input.

```python
class TITANFixedEpitopeWrapper:
    def __init__(self, titan_torch, device, epitope_fpath):
        self._device = device
        self._titan_model = titan_torch
        # load the epitope from file
        self._epitope = torch.IntTensor(load_data(epitope_fpath)).to(device)

    def predict(self, x: np.array):
        self._titan_model.eval() # eval mode for torch models
        ligand = self._epitope.repeat(len(x), 1) # adjusts the epitope to the right size
        ...
            # turns the np.array to torch tensor
            receptors = torch.FloatTensor(x).to(self._device)
        ...
            # runs the prediction using the torch forward function
            # TITAN returns more than an output, the predictions are the first ones
            pred = self._titan_model(ligand, receptors)[0]
            
        # threshold to get labels, and turn results to numpy array
        return (pred[:, 0] > 0.5).int().detach().cpu().numpy()
```

NOTE that we are doing this because the epitope is fixed and its more memory-efficient to just move around the TCRs. 
However, if the epitope was not fixed, it could be possible to also pass the TCR and the epitope as a single array (if dimensions allow it), and then split them in the predict function.

Unfortunately, the original data format of the TCR is not categorical as it is expected by the `Anchors` algorithm.
In our scenario, this can be easily solved. The original TCR input for `TITAN` is 3D: we can use this fact to differentiate when
the predict function is used within `Anchors` vs. the rest of the pipeline.

```python
class TITANFixedEpitopeWrapper:
    ...

    def predict(self, x: np.array):
        ...
        if len(x.shape) == 3:
            # we are dealing with the original input
            ...
        else:
            # the predict is being called within anchors
            ...
            ## .. some preprocessing to turn the categorical inputs to the original format expected by TITAN
        
        ...
        # threshold to get labels, and turn results to numpy array
        return (pred[:, 0] > 0.5).int().detach().cpu().numpy()
```

We are now ready to implement the model loader.

## Implement the model loader

The model loader is just a routine to, well, load a model. 
The expected interface is

```python
def your_model_loader(model_config, *args, **kwargs):
    # instantiate your model
    ...
    return model
```

`model_config` is a dictionary containing the parameters to construct your model. 

Now that you implemented the loader, you just need to add it to the model registry.
To do this, simply add the model loader in `model.py`:

```python
...

MODEL_LOADERS = {
    'your_model_loader_id': your_model_loader
}
```

The identifier could be anything you like that is not already in use. 
Now you can use your model in the pipeline: in your config file, you just need to set the entries

```json
  "model": {
    "model_id": "your_model_loader_id",
    "model_params": { # this is the dict passed to the model loader
        ... # info to build your model
    }
  }
```
