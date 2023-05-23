from torch import nn
from train_base import train
from model._vgg import _vgg

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
        model = _vgg("D", batch_norm=False, init_weights=True, num_classes=10)
        log_file_name = "vgg_no_batch_norm.log"
        train(model, log_file_name)
    elif args.model_type == 2:
        #####################
        ## Full Batch Norm ##
        #####################
        model = _vgg("D", batch_norm=True, init_weights=True, num_classes=10)
        log_file_name = "vgg_batch_norm.log"
        if args.clamp_value != -1:
            train(model, log_file_name, clamp_value=args.clamp_value)
        else:
            train(model, log_file_name)
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
