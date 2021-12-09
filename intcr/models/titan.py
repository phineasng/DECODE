import torch
from paccmann_predictor.models import MODEL_FACTORY


def load_titan(model_config, device=torch.device('cpu')):
    model_params = model_config['params']
    model = MODEL_FACTORY['bimodal_mca'](model_params).to(device)
    return model
