import pdb
import torch
import numpy as np
import torchvision.transforms as transforms
import scipy as sp
import scipy.stats

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])

# RelationNet2: Xueting Zhang
image_transform = transforms.Compose([
            transforms.Resize(96), # 256->224
            transforms.CenterCrop(84),
            transform
])

def LoadParameter(_structure, _parameterDir=None):
    if _parameterDir:
        model_state_dict = _structure.state_dict()
        for key in _parameterDir:
            model_state_dict[key.replace('module.', '')] = _parameterDir[key]
        _structure.load_state_dict(model_state_dict)

    return _structure

def accuracy(predicts, targets):
    """Computes the precision@k for the specified values of k"""
    if len(predicts) != len(targets):
        raise ValueError(f'len(output_pred) == len(target)')
    rewards = [1 if item_predict == item_target else 0 for item_predict, item_target in zip(predicts, targets)]
    return np.sum(rewards)/float(len(rewards))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h