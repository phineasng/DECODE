import torch
import numpy as np
from intcr.pipeline.utils import load_data
from paccmann_predictor.models import MODEL_FACTORY


class TITANFixedEpitopeWrapper:
    def __init__(self, titan_torch, device, epitope_fpath):
        self._device = device
        self._titan_model = titan_torch
        self._epitope = torch.IntTensor(load_data(epitope_fpath)).to(device)

    def predict(self, x: np.array):
        receptors = torch.FloatTensor(x).to(self._device)
        ligand = self._epitope.repeat(len(x), 1)
        pred = self._titan_model(ligand, receptors)
        print(pred.shape)
        return pred


def load_titan_fixed_epitope(model_config, device=torch.device('cpu')):
    model_params = model_config['params']
    model_ckpt = model_config['ckpt']
    model = MODEL_FACTORY['bimodal_mca'](model_params).to(device)
    model.load(model_ckpt, map_location=device)
    model = TITANFixedEpitopeWrapper(model, device, model_config['fixed_epitope_path'])
    return model

