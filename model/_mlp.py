import torch

from torch import nn
from typing import Union, List, Any, cast
from model.modified_layer import ModifiedLinear 

class MLP(nn.Module):
    def __init__(self, 
                 in_features: int,
                 cfg: List[Union[str, int]], 
                 batch_norm: bool = False,
                 num_classes: int = 1000) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        _in_features = in_features
        layers += [ModifiedLinear(_in_features, cfg[0]), nn.BatchNorm1d(cfg[0]), nn.ReLU(inplace=True)]
        _in_features = cfg[0]
        for v in cfg[1:]:
            v = cast(int, v)
            linear_layer = ModifiedLinear(_in_features, v)
            if batch_norm:
                layers += [linear_layer, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [linear_layer, nn.ReLU(inplace=True)]
            _in_features = v
        self.features_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            ModifiedLinear(_in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            ModifiedLinear(32, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

def _mlp(in_features: int, cfg: List[Union[int, str]], batch_norm: bool, num_classes: int, **kwargs: Any) -> MLP:
    model = MLP(in_features, cfg, batch_norm, num_classes)
    return model
