"""
@time: 2026/02/12
@file: train_engine.py
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
from Classification.utils import LOGGER
from Classification.utils import misc

def train_one_epoch(model, optimizer, data_loader, loss_scaler, device, epoch):
    model.train()
    lr = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)
    total_acc = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    loss_function = torch.nn.CrossEntropyLoss()

    batch_num = len(data_loader)
    sample_num = len(data_loader.dataset)
    # LOGGER.info(f"batch_num: {batch_num}; sample_num: {sample_num}")
    if misc.is_main_process():
        data_loader = tqdm(data_loader)

    optimizer.zero_grad()
    torch.cuda.empty_cache()
    for step, data in enumerate(data_loader):
        images = data['img']
        labels = data['cls']
        with torch.no_grad():
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
        # 正向传播
        output = model(images)

        loss = loss_function(output, labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            LOGGER.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        # 反向传播
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)

        torch.cuda.synchronize()

        with torch.no_grad():
            preds = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = preds.eq(labels.view_as(preds)).sum()
            samples_num = labels.size(0)
            acc = correct.float() / samples_num
            total_acc += acc
            total_loss += loss
            total_correct += correct
        # Synchronize metrics across GPUs
        loss_value_reduce = misc.all_reduce_mean(loss)
        acc_value_reduce = misc.all_reduce_mean(acc)

        # samples_num = misc.all_reduce(samples_num)
        lr = optimizer.param_groups[0]["lr"]
        if misc.is_main_process():
            data_loader.desc = "[train epoch {}] loss: {:.4f}, Acc: {:.4f}, lr: {:.5f}".format(
                epoch, loss_value_reduce, acc_value_reduce, lr
            )
    # Synchronize metrics across GPUs
    total_loss = misc.all_reduce(total_loss)
    total_acc = misc.all_reduce(total_acc)
    batch_num = misc.all_reduce(batch_num)
    total_correct = misc.all_reduce(total_correct)
    # sample_num = misc.all_reduce(sample_num)

    avg_loss = total_loss / batch_num
    avg_accuracy = total_acc / batch_num
    avg_accuracy_true = total_correct * 1.0 / sample_num

    if misc.is_main_process():
        LOGGER.info("Epoch {}: Average Loss: {:.4f}, Average Accuracy: {:.4f}, Accuracy: {:.4f}, lr: {:.5f}".format(
            epoch, avg_loss, avg_accuracy, avg_accuracy_true, lr
        ))
    return avg_loss, avg_accuracy_true, lr