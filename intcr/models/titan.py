import torch
from paccmann_predictor.models import MODEL_FACTORY


class TITANFixedEpitopeWrapper:
    def __init__(self, titan_torch, epitope):
        self._titan_model = titan_torch
        self._epitope = epitope

    def predict(self, x):
        pass


def load_titan(model_config, device=torch.device('cpu')):
    model_params = model_config['params']
    model_ckpt = model_config['ckpt']
    model = MODEL_FACTORY['bimodal_mca'](model_params).to(device)
    model.load(model_ckpt, map_location=device)
    return model

