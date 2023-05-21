import torch

from torch import nn
from torch import Tensor
from typing import Optional, Callable, Type, Union, List, Any, cast, Dict
from model.modified_layer import ModifiedConv2d, ModifiedMaxPool2d, ModifiedLinear, ModifiedAdaptiveAvgPool2d

class MLP(nn.Module):
    def __init__(self, 
                 in_features: int,
                 cfg: List[Union[str, int]], 
                 batch_norm: bool = False,
                 num_classes: int = 1000):
        layers: List[nn.Module] = []
        _in_features = in_features
        for v in cfg:
            v = cast(int, v)
            linear_layer = ModifiedLinear(_in_features, v)
            if batch_norm:
                layers += [linear_layer, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [linear_layer, nn.ReLU(inplace=True)]
            _in_features = v
        self.features_extractor = nn.Sequential(*layers)
        self.classifier = ModifiedLinear(_in_features, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

def _mlp(in_features: int, cfg: List[Union[int, str]], batch_norm: bool, num_classes: int, **kwargs: Any) -> MLP:
    model = MLP(in_features, cfg, batch_norm, num_classes)
    return model
