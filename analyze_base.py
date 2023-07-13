import os
import torch
import re
import numpy as np
import argparse

from constant import SELECTED_LAYERS, VISUALIZE_LAYERS
from matplotlib import pyplot as plt
from utils import default_config, add_dict_to_argparser

def analyze(batch_norm, model):
    selected_layers = SELECTED_LAYERS[model]
    visualize_layers = VISUALIZE_LAYERS[model]
    if not os.path.isdir("image"):
        os.mkdir("image")
    file_names = os.listdir("variance")
    variance = {}
    for file in file_names:
        _, epoch, iters = file.split(".")[0].split("_")
        if int(epoch) > 60:
            continue
        all_iter = int(epoch) * 782 + int(iters)
        data = torch.load(f"variance/{file}")
        mean_var = []
        var_var = []
        for each in data:
            mean_var.append(torch.mean(each.view(-1)).item())
            var_var.append(torch.var(each.view(-1)).item())
        mean_var = [mean_var[i] for i in selected_layers]
        var_var = [var_var[i] for i in selected_layers]
        variance[all_iter] = {
            "mean": mean_var,
            "var": var_var
        }

    variance = dict(sorted(variance.items()), key=lambda x: x[0])

    list_of_mean = [{}, {}, {}, {}]
    list_of_var = [{}, {}, {}, {}]

    for i, key in enumerate(variance.keys()):
        if i % 5 == 0:
            for j in range(4):
                list_of_mean[j][key] = variance[key]['mean'][visualize_layers[j]]
                list_of_var[j][key] = np.sqrt(variance[key]['var'][visualize_layers[j]])

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('minibatch', fontdict={'fontsize': 18})
    ax.set_ylabel(r'$\sigma^{2}$', fontdict={'fontsize': 18})
    for i in range(4):
        ax.plot(list(list_of_mean[i].keys()), list(list_of_mean[i].values()), label=f"{batch_norm}, Layer {visualize_layers[i] + 1}")
        ax.fill_between(list(list_of_mean[i].keys()), np.clip(np.array(list(list_of_mean[i].values())) - np.array(list(list_of_var[i].values())), 0, None), np.array(list(list_of_mean[i].values())) + np.array(list(list_of_var[i].values())), alpha=0.3)
    ax.legend(fontsize=12, loc="upper left")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.tight_layout()
    if batch_norm == "BN":
        plt.savefig(f"image/{model}_{batch_norm}.png", dpi=250)
    elif batch_norm == "GN":
        plt.savefig(f"image/{model}_{batch_norm}.png", dpi=250)
    elif batch_norm == "LN":
        plt.savefig(f"image/{model}_{batch_norm}.png", dpi=250)
    else:
        plt.savefig(f"image/{model}_without_batch_norm.png", dpi=250)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    args = parser.parse_args()

    batch_norm = ""
    log_file = ""
    if args.model_type == 0:
        batch_norm = "w/o BN"
        log_file = f"{args.model}_wo_norm.txt"
    elif args.model_type == 2:
        if args.norm_type == "batch":
            batch_norm = "BN"
            log_file = f"{args.model}_batch.txt"
        elif args.norm_type =="group":
            batch_norm = "GN"
            log_file = f"{args.model}_group.txt"
        elif args.norm_type == "layer":
            batch_norm = "LN"
            log_file = f"{args.model}_layer.txt"
    else:
        raise NotImplementedError()

    analyze(batch_norm, args.model)
