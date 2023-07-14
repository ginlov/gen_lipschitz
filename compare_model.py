import os
import torch
import re
import argparse

from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from utils import default_config, add_dict_to_argparser
from constant import COMPARE_LAYERS


def analyze_log(log):
    acc1 = []
    acc5 = []
    loss = []
    test_acc1 = []
    test_acc5 = []

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
                
    def analysis_test(log_row):
        log_row = re.sub(" +", " ", log_row)
        test_acc1.append(log_row.split()[-3])
        test_acc5.append(log_row.split()[-1])

    for log_row in log:
        if "Epoch" in log_row:
            if int(log_row.split("]")[0].split("[")[1]) > 40:
                break
            analysis_train(log_row)
        elif "*" in log_row:
            analysis_test(log_row)
    acc1 = [float(acc1[i]) for i in range(320) if i % 8 == 7]
    acc5 = [float(acc5[i]) for i in range(320) if i % 8 == 7]
    loss = [float(loss[i]) for i in range(320) if i % 8 == 7]

    return acc1, acc5, loss, test_acc1[:40], test_acc5[:40]


def draw_image_acc5(
        acc5, 
        test_acc5, 
        acc5_wo_norm, 
        test_acc5_wo_norm,
        norm_type,
        model
    ):
    assert len(acc5) == len(test_acc5) == len(acc5_wo_norm) == len(test_acc5_wo_norm)
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Epoch", fontdict={'fontsize': 18})
    ax.set_ylabel("Accuracy", fontdict={'fontsize': 18})

    major_yticks = list(range(0, 110, 10))
    ax.set_yticks(major_yticks)
    ax.set_yticks(major_yticks)

    # Draw images
    num_epoch = len(acc5)
    plt.plot(list(range(num_epoch)), [float(item) for item in acc5], 'b--', label=f"{norm_type}, train")
    plt.plot(list(range(num_epoch)), [float(item) for item in test_acc5], color='b',  label=f"{norm_type}, test")
    plt.plot(list(range(num_epoch)), [float(item) for item in acc5_wo_norm], 'r--', label=f"w/o {norm_type}, train")
    plt.plot(list(range(num_epoch)), [float(item) for item in test_acc5_wo_norm], color='r', label=f"w/o {norm_type}, test")
    ax.legend(fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.tight_layout()
    plt.savefig(f"train_test_accuracy_{model}_{norm_type}.png", dpi=250)


def compare_acc(
        log_wo_norm, 
        log_norm, 
        model, 
        norm_type
    ):
    acc1, acc5, loss, test_acc1, test_acc5 = analyze_log(log_norm)
    acc1_wo, acc5_wo, loss_wo, test_acc1_wo, test_acc5_wo = analyze_log(log_wo_norm)
    draw_image_acc5(
        acc5=acc5,
        test_acc5=test_acc5,
        acc5_wo_norm=acc5_wo,
        test_acc5_wo_norm=test_acc1_wo,
        norm_type=norm_type,
        model=model
    )


def variance_compare(
        model,
        norm_type,
        log_folder,
        log_folder_wo_norm
):
    variance_norm = torch.load(os.path.join(log_folder, "variance", "variance_39_699.pth"), map_locaiton="cpu")
    variance_wo_norm = torch.load(os.path.join(log_folder_wo_norm, "variance", "variance_39_699.pth"),
                                  map_location="cpu")
    draw_variance(
        variance_norm,
        variance_wo_norm,
        model,
        norm_type
    )


def draw_variance(
    variance_norm,
    variance_wo_norm,
    model,
    norm_type
):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Layer", fontdict={'fontsize': 18})
    ax.set_ylabel(r"$\sigma^{2}$", fontdict={'fontsize': 18})
    major_yticks = [10e-1, 10e0, 10, 10e2]
    ax.set_yticks(major_yticks)
    bn = []
    no_bn = []
    ax.set_yscale("log")
    selected_index = COMPARE_LAYERS[model]

    for index, each_variance in enumerate(variance_norm):
        if index in selected_index:
            temp = ax.boxplot(each_variance.numpy(), widths=0.5, positions=[index+1], whiskerprops=dict(color="blue"),
                              capprops=dict(color="blue"), medianprops=dict(color="blue"), boxprops=dict(color="blue"),
                              flierprops=dict(markeredgecolor="blue"))
            bn.append(temp["boxes"][0])
    for index, each_variance in enumerate(variance_wo_norm):
        if index in selected_index:
            temp = ax.boxplot(each_variance.numpy(), widths=0.5, positions=[index+1], whiskerprops=dict(color="red"),
                              capprops=dict(color="red"), medianprops=dict(color="red"), boxprops=dict(color="red"),
                              flierprops=dict(markeredgecolor="red"))
            no_bn.append(temp["boxes"][0])

    ax.legend([tuple(bn), tuple(no_bn)], [norm_type, f"w/o {norm_type}"], fontsize=12,
              handler_map={tuple: HandlerTuple(ndivide=None)}, loc="lower left")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig.tight_layout()
    plt.savefig(f"image/variance_fig_{model}_{norm_type}.png", dpi=250)


def gamma_variance_correlation():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_config())

    args = parser.parse_args()

    log_file = "_".join([args.model, args.norm_type]) + ".txt"
    log_folder = "_".join([args.model, args.norm_type])
    log_wo_norm = f"{args.model}_wo_norm.txt"
    log_folder_wo_norm = "_".join([args.model, "wo_norm"])

    # draw accuracy
    compare_acc(log_wo_norm=log_wo_norm, log_norm=log_file, model=args.model, norm_type=args.norm_type)
