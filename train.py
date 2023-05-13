from model._resnet import _resnet, BasicBlock
from torchvision import datasets, transforms
from enum import Enum
from torch import nn
from loguru import logger

import torch
import time
import shutil
import argparse
import numpy as np


def train(model, log_file_name=""):
    ##############################
    ###### Settings ##############
    ##############################
    global best_acc1
    batch_size = 64
    workers = 5
    num_epoch = 90
    lr = 1e-3
    weight_decay = 1e-4
    momentum = 0.9

    ###############################
    ### LOADING DATASET ###########
    ###############################
    logger.info("Loading data")
    # traindir = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"
    # valdir = "/kaggle/working/imagenet-object-localization-challenge/val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    train_dataset = datasets.CIFAR10(root="cifar_train", train=True, 
                                     transform=transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                     ]),
                                     download=True)
    val_dataset = datasets.CIFAR10(root="cifar_val", train=False, 
                                   transform=transforms.Compose([
                                   # transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   normalize,
                                   ]),
                                   download=True)

    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    ###################################
    ## LOADING COMPULSORY COMPONENTS ##
    ###################################
    # model = _resnet(BasicBlock, [2, 2, 2, 2])
    logger.info("Preparing model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    logger.info("Start training")
    for i in range(num_epoch):
        train_epoch(model, train_loader, optimizer, loss_fn, device, log_file_name, i)

        acc1, acc5 = validate_epoch(model, val_loader, loss_fn, device, log_file_name, i)

        scheduler.step() 

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint(
            {
                'epoch': i + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            is_best
        )


def train_epoch(model, train_loader, optimizer, loss_fn, device, log_file, epoch):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter('Loss', ':.4e')
    weight_norm = AverageMeter('Weight Norm', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, weight_norm, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
        log_file_name=log_file)

    model.train()
    end = time.time()

    for i, (image, target) in enumerate(train_loader):
        ########################
        ## LOADING DATA TIME ###
        ########################
        data_time.update(time.time() - end)

        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(image)

        loss = loss_fn(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        weight_norm_value = cal_weight_norm(model, norm=2)
        losses.update(loss.item(), image.size(0))
        weight_norm.update(weight_norm_value, 1)
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.display(i + 1)


def validate_epoch(model, valid_loader, loss_fn, device, log_file, epoch):
    
    def run_validate(valid_loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (image, target) in enumerate(valid_loader):
                i = base_progress + i
                image = image.to(device, non_blocking=True)
                target = image.to(device, non_blocking=True)
                output = model(image)
                loss = loss_fn(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), image.size(0))
                top1.update(acc1[0], image.size(0))
                top5.update(acc5[0], image.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 100 == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(valid_loader), 
        [batch_time, losses, top1, top5],
        prefix='Test: ',
        log_file_name=log_file)

    model.eval()
    run_validate(valid_loader)
    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=int, default=0)
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
        model = _resnet(BasicBlock, [2, 2, 2, 2], norm_layer=None)
        log_file_name = "two_batch_norm.log"
        train(model, log_file_name)
    elif args.model_type == 2:
        #####################
        ## Full Batch Norm ##
        #####################
        model = _resnet(BasicBlock, [2, 2, 2, 2], norm_layer=nn.BatchNorm2d)
        log_file_name = "batch_norm.log"
        train(model, log_file_name)
    else:
        raise NotImplementedError()


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_file_name=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_file = log_file_name

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        with open(self.log_file, "a+") as f:
            f.write("\t".join(entries) + "\n")
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))
        with open(self.log_file, "a+") as f:
            f.write(" ".join(entries) + "\n")


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cal_weight_norm(model, norm=2):
    def process_layer(layer, norm, cum_prod = 1.0):
        children = list(layer.children())
        if len(children) > 0:
            for each in children:
                process_layer(each, norm, cum_prod)
        elif hasattr(layer, "weight"):
            cum_prod *= np.log10(np.linalg.norm(layer.weight.detach().cpu().numpy().reshape(-1), ord=norm))
    cum_prod = 1.0
    process_layer(model, norm, cum_prod)
    return cum_prod

if __name__ == "__main__":
    main()
