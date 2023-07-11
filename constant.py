from model._resnet import BasicBlock, _resnet
from model._vgg import _vgg
from model._mlp import _mlp

SELECTED_LAYERS = {
    "resnet": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "vgg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "mlp": [0, 1, 2, 3, 4, 5, 6, 7]
}

VISUALIZE_LAYERS = {
    "resnet": [4, 9, 14, 19],
    "vgg": [0, 4, 9, 12],
    "mlp": [0, 2, 4, 7]
}

MODEL_CONFIG = {
    "resnet": {
        "block": BasicBlock,
        "layers": [2, 2, 2, 2],
        "norm_layer": None,
        "num_classes": 10
    },
    "vgg": {
        "cfg": "D",
        "norm_layer": None,
        "init_weights": True,
        "num_classes": 10
    },
    "mlp": {
        "in_features": 3 * 32 * 32,
        "cfg": [1024, 1024, 512, 512, 256, 256, 128, 64],
        "norm_layer": None,
        "num_classes": 10
    }
}

MODEL_MAP = {
    "resnet": _resnet,
    "mlp": _mlp,
    "vgg": _vgg
}
