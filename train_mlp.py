from train_base import train
from model._mlp import _mlp

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=int, default=0)
    parser.add_argument("--clamp_value", type=float, default=-1)
    args = parser.parse_args()

    # hidden_layer = [1024, 1024, 512, 512, 512, 256, 256, 128, 128, 64]
    hidden_layer = [1024, 1024, 512, 512, 256, 256, 128, 64]
    # hidden_layer = [200, 200, 200, 200, 200, 200, 200, 200]

    # hidden_layer = [512] * 10

    if args.model_type == 0:
        ###################
        ## No Batch Norm ##
        ###################
        model = _mlp(3 * 32 * 32, cfg=hidden_layer, batch_norm=False, num_classes=10)
        log_file_name = "mlp_no_batch_norm.log"
        train(model, log_file_name)
    elif args.model_type == 2:
        #####################
        ## Full Batch Norm ##
        #####################
        model = _mlp(3 * 32 * 32, cfg=hidden_layer, batch_norm=True, num_classes=10)
        log_file_name = "mlp_batch_norm.log"
        if args.clamp_value != -1:
            train(model, log_file_name, clamp_value=args.clamp_value)
        else:
            train(model, log_file_name)
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
