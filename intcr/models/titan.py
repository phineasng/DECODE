import torch
from paccmann_predictor.models import MODEL_FACTORY


def load_titan(model_config, device=torch.device('cpu')):
    model_params = model_config['params']
    model_ckpt = model_config['ckpt']
    model = MODEL_FACTORY['bimodal_mca'](model_params).to(device)
    model.load(model_ckpt, map_location=device)
    return model
