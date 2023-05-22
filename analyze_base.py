import os
import torch
import re
import numpy as np
import argparse

from constant import SELECTED_LAYERS, VISUALIZE_LAYERS
from matplotlib import pyplot as plt

def analyze(log_file, batch_norm, model):
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
    else:
        plt.savefig(f"image/{model}_without_batch_norm.png", dpi=250)


def analyze_log(log):
    acc1 = []
    acc5 = []
    loss = []
    def analysis_train(log_row):
        items = log_row.split("\t")
        for item in items:
            item = re.sub(" +", " ", item)
            if "Acc@1" in item:
                acc1.append(item.split(" ")[1])
            elif "Acc@5" in item:
                acc5.append(item.split(" ")[1])
            elif "Loss" in item:
                loss.append(item.split(" ")[1])
                
    test_acc1 = []
    test_acc5 = []
    def analysis_test(log_row):
        log_row = re.sub(" +", " ", log_row)
        test_acc1.append(log_row.split()[-3])
        test_acc5.append(log_row.split()[-1])

    for log_row in log:
        if "Epoch" in log_row:
            if int(log_row.split("]")[0].split("[")[1]) > 60:
                break
            analysis_train(log_row)
        elif "*" in log_row:
            analysis_test(log_row)
    acc1 = [float(acc1[i]) for i in range(488) if i % 8 == 7]
    acc5 = [float(acc5[i]) for i in range(488) if i % 8 == 7]
    loss = [float(loss[i]) for i in range(488) if i % 8 == 7]

    return acc1, acc5, loss, test_acc1, test_acc5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--model_type", type=int, default=0)

    args = parser.parse_args()

    model = ""
    if args.model == 0:
        model = "mlp"
    elif args.model == 1:
        model = "vgg"
    elif args.model == 2:
        model = "resnet"
    else:
        raise NotImplementedError()

    batch_norm = ""
    log_file = ""
    if args.model_type == 0:
        batch_norm = "w/o BN"
        log_file = f"{model}_no_batch_norm.log"
    elif args.model_type == 2:
        batch_norm = "BN"
        log_file = f"{model}_batch_norm.log"
    else:
        raise NotImplementedError()

    analyze(log_file, batch_norm, model)
