from train_base import train
from torch import nn
from typing import Dict
from model._mlp import _mlp
from model._resnet import _resnet
from model._vgg import _vgg
from constant import MODEL_CONFIG, MODEL_MAP

import argparse


def default_config():
    return {
        "model": "mlp",
        "model_type": 0,
        "clamp_value": -1.0,
        "norm_type": "batch"
    }

def add_dict_to_argparser(
    parser: argparse.ArgumentParser,
    config_dict: Dict
    ):
    for k, v in config_dict.items():
        v_type = type(v)
        parser.add_argument(f"--{k}", type=v_type, default=v)

def start_train(
        args: argparse.Namespace,
        debug:bool=False
    ):

    ## create model
    config = MODEL_CONFIG[args.model]
    if args.model_type == 2:
        if args.norm_type== "batch":
            if args.model == "mlp":
                config["norm_layer"] = nn.BatchNorm1d
            else:
                config["norm_layer"] = nn.BatchNorm2d
        elif args.norm_type == "group":
            config["norm_layer"] = nn.GroupNorm
        elif args.norm_type == "layer":
            config["norm_layer"] = nn.LayerNorm
        else:
            raise NotImplementedError("This norm type has not been implemented yet.")
    
    model = MODEL_MAP[args.model](**config)


    ## training
    if args.model_type == 0:
        norm = "wo_norm"
    else:
        norm = args.norm_type
    log_file_name = "_".join([args.model, norm]) + ".txt" 
    training_config = {
        "model": model,
        "log_file_name": log_file_name, 
        "clamp_value": args.clamp_value
    }

    if debug:
        return config, training_config

    train(**training_config)


def main():
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())
    
    args = parser.parse_args()

    start_train(args, False)
