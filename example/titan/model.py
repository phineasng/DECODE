import torch
import numpy as np
from intcr.pipeline.utils import load_data
from paccmann_predictor.models import MODEL_FACTORY
from data import BLOSUM_IDX2KEY, BLOSUM62


class TITANFixedEpitopeWrapper:
    def __init__(self, titan_torch, device, epitope_fpath):
        self._device = device
        self._titan_model = titan_torch
        self._epitope = torch.IntTensor(load_data(epitope_fpath)).to(device)

    def predict(self, x: np.array):
        self._titan_model.eval()
        ligand = self._epitope.repeat(len(x), 1)
        if len(x.shape) == 3:
            receptors = torch.FloatTensor(x).to(self._device)
        else:
            new_x = []
            for s in x:
                sample = []
                for token in s:
                    sample.append(BLOSUM62[BLOSUM_IDX2KEY[np.int(token)]])
                new_x.append(np.stack(np.array(sample), axis=0))
            receptors = torch.FloatTensor(np.stack(new_x, axis=0)).to(self._device)
        if len(x) == 1:
            ligand = ligand.repeat(2, 1)
            receptors = receptors.repeat(2, 1, 1)
            pred = self._titan_model(ligand, receptors)[0]
            pred = pred[0:1, :]
        else:
            pred = self._titan_model(ligand, receptors)[0]
        return (pred[:, 0] > 0.5).int().detach().cpu().numpy()


def load_titan_fixed_epitope(model_config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_params = model_config['params']
    model_ckpt = model_config['ckpt']
    model = MODEL_FACTORY['bimodal_mca'](model_params).to(device)
    model.load(model_ckpt, map_location=device)
    model = TITANFixedEpitopeWrapper(model, device, model_config['fixed_epitope_path'])
    return model


MODEL_LOADERS = {
    'titan_fixed_epitope': load_titan_fixed_epitope,
}