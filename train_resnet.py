from torch import nn
from train_base import train
from model._resnet import BasicBlock, _resnet

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=int, default=0)
    parser.add_argument("--clamp_value", type=float, default=-1)
    args = parser.parse_args()

    if args.model_type == 0:
        ###################
        ## No Batch Norm ##
        ###################
        model = _resnet(BasicBlock, [2, 2, 2, 2], norm_layer=None)
        log_file_name = "no_batch_norm.log"
        train(model, log_file_name)
    elif args.model_type == 1:
        ####################
        ## Two Batch Norm ##
        ####################
        model = _resnet(BasicBlock, [2, 2, 2, 2], norm_layer=None, signal=1)
        log_file_name = "two_batch_norm.log"
        train(model, log_file_name)
    elif args.model_type == 2:
        #####################
        ## Full Batch Norm ##
        #####################
        model = _resnet(BasicBlock, [2, 2, 2, 2], norm_layer=nn.BatchNorm2d)
        log_file_name = "batch_norm.log"
        if args.clamp_value != -1:
            train(model, log_file_name, clamp_value=args.clamp_value)
        else:
            train(model, log_file_name)
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
