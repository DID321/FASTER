"""
@time: 2026/02/12
@file: test_engine.py
@author: WD                     ___       __   ________
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ /
                                ____/|__/      /_____/

"""
import torch
import sys
from tqdm import tqdm
import math
import torch.distributed as dist
from torch.nn import functional as F
from Classification.utils import misc
from Classification.utils import LOGGER
from timm.utils import accuracy
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # 切换到评估模式
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)

    loss_function = torch.nn.CrossEntropyLoss()
    data_loader_pbar = tqdm(data_loader)
    batch_num = len(data_loader_pbar)
    with torch.no_grad():
        for step, data in enumerate(data_loader_pbar):
            images = data['img']
            labels = data['cls']
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
            # 正向传播
            output = model(images)

            loss = loss_function(output, labels)
            # torch.cuda.synchronize()
            preds = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = preds.eq(labels.view_as(preds)).sum()
            acc = correct.float() / labels.size(0)
            total_loss += loss
            total_correct += correct
            # Synchronize metrics across GPUs
            # loss_value_reduce = misc.all_reduce_mean(loss)

            if misc.is_main_process():
                data_loader_pbar.desc = "[valid epoch {}] loss: {:.4f}, Acc: {:.4f} ".format(
                    epoch, loss.item(), acc.item()
                )
    # Synchronize metrics across GPUs
    # total_loss = misc.all_reduce(total_loss)
    # batch_num = misc.all_reduce(batch_num)

    avg_loss = total_loss / batch_num
    avg_acc = total_correct.float() / len(data_loader.dataset)

    if misc.is_main_process():
        LOGGER.info("Epoch {} ---valid : Average Loss: {:.4f}, Accuracy: {:.4f} ".format(
            epoch, avg_loss.item(), avg_acc.item()
        ))
    return avg_loss.item(), avg_acc.item()

@torch.no_grad()
def model_test(model, data_loader, device):
    # 切换到评估模式
    model.eval()
    total_correct = torch.tensor(0, device=device)
    total_outputs = []
    total_labels = []

    data_loader_pbar = tqdm(data_loader)
    batch_num = len(data_loader_pbar)
    with torch.no_grad():
        for step, data in enumerate(data_loader_pbar):
            images = data['img']
            labels = data['cls']
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
            # 正向传播
            output = model(images)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            # torch.cuda.synchronize()
            preds = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            correct = preds.eq(labels.view_as(preds)).sum()
            acc = correct.float() / labels.size(0)
            total_correct += correct
            total_outputs.append(output)
            total_labels.append(labels)
            # Synchronize metrics across GPUs
            # loss_value_reduce = misc.all_reduce_mean(loss)

            data_loader_pbar.desc = "[Test {} / {}] Acc@1: {:.4f} Acc@5: {:.4f}".format(step, batch_num, acc1.item(), acc5.item())
    # Synchronize metrics across GPUs
    # total_loss = misc.all_reduce(total_loss)
    # batch_num = misc.all_reduce(batch_num)
    total_outputs = torch.cat(total_outputs, dim=0)
    ground_truth = torch.cat(total_labels, dim=0)
    total_acc1, total_acc5 = accuracy(total_outputs, ground_truth, topk=(1, 5))
    total_preds = total_outputs.max(1)[1]
    avg_acc = total_correct.float() / len(data_loader.dataset)


    LOGGER.info("✅ Test Complete! --- Average Accuracy@1: {:.4f} Accuracy@5: {:.4f} ".format(total_acc1.item(), total_acc5.item()))
    return total_preds.cpu().numpy(), total_acc1.item(), total_acc5.item()
