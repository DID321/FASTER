import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy
from tqdm import tqdm
import Pretrain.utils.misc as misc
import Pretrain.utils.lr_sched as lr_sched
from Pretrain.utils import LOGGER

def train_one_epoch(model: torch.nn.Module, model_ema, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    lr = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)
    total_acc = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)

    accum_iter = args.accum_iter
    if misc.is_main_process():
        data_loader = tqdm(data_loader)

    batch_num = len(data_loader)
    sample_num = len(data_loader.dataset)

    optimizer.zero_grad()
    torch.cuda.empty_cache()

    for step, data in enumerate(data_loader):
        images = data['img']
        labels = data['cls']
        with torch.no_grad():
            if device is not None:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
        # we use a per iteration (instead of per epoch) lr scheduler
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            LOGGER.warning("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)
        # torch.cuda.synchronize()

        with torch.no_grad():
            preds = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = preds.eq(labels.view_as(preds)).sum()
            samples_num = labels.size(0)
            acc = correct.float() / samples_num
            total_acc += acc
            total_loss += loss
            total_correct += correct

        lr = optimizer.param_groups[0]["lr"]

        # Synchronize metrics across GPUs
        acc_value_reduce = misc.all_reduce_mean(acc)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        if misc.is_main_process():
            data_loader.desc = "[train epoch {}] loss: {:.4f}, Acc: {:.4f}, lr: {:.5f}".format(
                epoch, loss_value_reduce, acc_value_reduce, lr
            )

    # Synchronize metrics across GPUs
    total_loss = misc.all_reduce(total_loss)
    total_acc = misc.all_reduce(total_acc)
    batch_num = misc.all_reduce(batch_num)
    total_correct = misc.all_reduce(total_correct)
    sample_num = misc.all_reduce(sample_num)


    avg_loss = total_loss / batch_num
    avg_accuracy = total_acc / batch_num
    avg_accuracy_true = total_correct.float() / sample_num

    if misc.is_main_process():
        LOGGER.info("Epoch {}: Average Loss: {:.4f}, Average Accuracy: {:.4f}, Accuracy: {:.4f}, lr: {:.5f}".format(
            epoch, avg_loss, avg_accuracy, avg_accuracy_true, lr
        ))
    return avg_loss, avg_accuracy_true, lr



@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # 切换到评估模式
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_correct = torch.tensor(0, device=device)
    total_preds = []
    total_labels = []
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
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))

            preds = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            correct = preds.eq(labels.view_as(preds)).sum()
            acc = correct.float() / labels.size(0)


            total_loss += loss
            total_correct += correct
            total_preds.append(output)
            total_labels.append(labels)
            # Synchronize metrics across GPUs
            # loss_value_reduce = misc.all_reduce_mean(loss)

            if misc.is_main_process():
                data_loader_pbar.desc = "[valid epoch {}] loss: {:.4f}, Acc1: {:.4f}, Acc5: {:.4f} ".format(
                    epoch, loss.item(), acc.item(), acc5.item()
                )
    # Synchronize metrics across GPUs
    # total_loss = misc.all_reduce(total_loss)
    # batch_num = misc.all_reduce(batch_num)
    total_preds = torch.cat(total_preds, dim=0)
    ground_truth = torch.cat(total_labels, dim=0)
    # ground_truth = torch.tensor([lb['cls'] for lb in data_loader.labels])
    total_acc1, total_acc5 = accuracy(total_preds, ground_truth, topk=(1, 5))
    avg_loss = total_loss / batch_num
    avg_acc = total_correct.float() / len(data_loader.dataset)

    if misc.is_main_process():
        LOGGER.info("Epoch {} ---valid : Average Loss: {:.4f}, Accuracy Top1: {:.4f} Accuracy Top5: {:.4f}".format(
            epoch, avg_loss.item(), avg_acc.item(), total_acc5.item()
        ))
    return avg_loss.item(), total_acc1.item(), total_acc5.item()


# @torch.no_grad()
# def evaluate(data_loader, model, device):
#     criterion = torch.nn.CrossEntropyLoss()
#
#     metric_logger = misc.MetricLogger(delimiter="  ")
#     header = 'Test:'
#
#     # switch to evaluation mode
#     model.eval()
#
#     for batch in metric_logger.log_every(data_loader, 10, header):
#         images = batch[0]
#         target = batch[-1]
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)
#
#         # compute output
#         with torch.cuda.amp.autocast():
#             output = model(images)
#             loss = criterion(output, target)
#
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#
#         batch_size = images.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
#         metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#           .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
#
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}