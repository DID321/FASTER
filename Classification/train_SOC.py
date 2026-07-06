"""
@time: 2026/02/09
@file: train_SOC.py
@author: WD                     ___       __   ________
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ /
                                ____/|__/      /_____/

"""
import sys
import torch
import numpy as np
import re
import os
import time
import pathlib
# 将项目根目录加入 Python 路径
ROOT = pathlib.Path(__file__).parent.parent
# print(ROOT)
sys.path.insert(0, str(ROOT))
import argparse
import utils.misc as misc
import timm.optim.optim_factory as optim_factory
import collections
from data.datasets import SARDataSet
from utils import yaml_load, show_config
from utils.callbacks import LogWriter
from models.resnet import ResNet_18, ResNet_34, ResNet_50, ResNet_101
from models.convnext import ConvNeXt_base, ConvNeXt_tiny, ConvNeXt_small, ConvNeXt_large 
from models.vgg import VGGNet
from models.hdanet import HDANet
from models.vit import vit_base_patch16
from models.hivit import hivit_base
from models.saratrx import saratr_x
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils import LOGGER
from utils.lr_sched import get_lr_scheduler, set_optimizer_lr
from utils.train_engine import train_one_epoch
from utils.test_engine import evaluate

# from utils.TrainTest import model_train, model_val, model_test
# from model.Model import convnext_1, ResNet_34

def parameter_setting():
    parser = argparse.ArgumentParser()
    # 训练迭代次数
    parser.add_argument('--epochs', type=int, default=200)
    # 训练batch_size
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # 冻结训练次数
    parser.add_argument('--freeze-epochs', type=int, default=0)
    # 优化器
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (AdamW, Adam, SGD)')
    # 学习率衰减策略
    parser.add_argument('--lr-decay-type', type=str, default='Cos', help='lr decay type (Step, Cos)')
    # 学习率
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # 动量
    parser.add_argument('--momentum', type=float, default=0.9)
    # 衰减速率
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    # log 文件夹
    parser.add_argument('--log-path', type=str, default='./Classification/logs/Convnext_SOC')
    # 保存权重文件夹
    parser.add_argument('--save-path', type=str, default='./Classification/logs/Convnext_SOC')
    # 训练图像尺寸
    parser.add_argument('--img-size', type=list, default=[224, 224])
    # 数据集配置文件所在根目录
    parser.add_argument('--data-cfg', type=str, default="./Classification/cfg/datasets/SOC.yaml")
    # 多少epoch保存一次权重
    parser.add_argument('--save-period', type=int, default=20)
    # 多少epoch验证一次
    parser.add_argument('--eval-period', type=int, default=1)
    # num_workers
    parser.add_argument('--num-workers', type=int, default=4)
    # 预训练权重路径，如果不想载入就设置为空字符 # ./weights/resnet18-f37072fd.pth
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    # 随机数种子
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 显卡选择
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--accum-iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # distributed training parameters
    # 所有参与训练的进程总数 通常等于 GPU 总数
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    # 用于指定该进程应该使用哪块 GPU
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--dist-on-itp', action='store_true')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--dist-url', default='file:///E:/SAR_Code/CSAR_ATR_Bench/temp_torch_dist_init', help='url used to set up distributed training')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parameter_setting()
    show_config(args.__dict__)
    # Setup distributed training
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    device = torch.device(args.device)
    LOGGER.info("✅ Using {} device.".format(device))
    # torch.cuda.manual_seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    history = collections.defaultdict(list)  # 记录每一折的各种指标

    cfg = yaml_load(args.data_cfg)
    # 记得修改cfg["test_txt_path"]
    train_dataset = SARDataSet(data_root=cfg["data_path"], img_txt_path=cfg["test_txt_path"], classes=cfg["names"], sar_config=cfg['sar_train'])
    test_dataset = SARDataSet(data_root=cfg["data_path"], img_txt_path=cfg["test_txt_path"], classes=cfg["names"], sar_config=cfg['sar_test'])

    if global_rank == 0 and args.log_path is not None and args.save_path is not None:
        os.makedirs(args.log_path, exist_ok=True)
        os.makedirs(args.save_path, exist_ok=True)
        log_writer = LogWriter(log_dir=args.log_path)
    else:
        log_writer = None

    if args.distributed:
        # 多卡分布式打开
        num_tasks = misc.get_world_size()
        # 单机单卡测试分布式
        # num_tasks = 2
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank,
                                                            shuffle=True)

    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    LOGGER.info("Sampler_train = %s" % str(train_sampler))
    LOGGER.info("Sampler_test = %s" % str(test_sampler))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=args.num_workers, collate_fn=SARDataSet.collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                                                  num_workers=args.num_workers, collate_fn=SARDataSet.collate_fn)

    LOGGER.info('train shape:{}, test shape:{}'.format(len(train_dataloader.dataset), len(test_dataloader.dataset)))

    # Initialize model
    # model = ResNet_18(num_classes=len(cfg["names"].keys()))
    # model = ConvNeXt_base(num_classes=len(cfg["names"].keys()))
    # model = VGGNet(num_classes=len(cfg["names"].keys()))
    # model = HDANet(num_classes=len(cfg["names"].keys()), in_ch=3)
    # model = vit_base_patch16(num_classes=len(cfg["names"].keys()))
    # model = hivit_base(num_classes=len(cfg["names"].keys()))
    model = saratr_x(num_classes=len(cfg["names"].keys()))
    # model = ResNet_34(num_classes=len(cfg["names"].keys()))
    # model = ResNet_50(num_classes=len(cfg["names"].keys()))
    # summary(model, input_data=[torch.randn((args.batch_size, 3, args.img_size[0], args.img_size[1]))], device=device)
    if args.weights != '':
        assert os.path.exists(args.weights), "⚠️ Warning! Weights file: '{}' not exist.".format(args.weights)
        model_dict = model.state_dict()
        checkpoint = torch.load(args.weights, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in checkpoint.items():
            k = 'model.' + k
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict, strict=False)
        LOGGER.info(f"✅ Successful Load Key: {str(load_key)[:500]} ……\nSuccessful Load Key Num: {len(load_key)}")
        LOGGER.info(f"❌ Fail To Load Key: {str(no_load_key)[:500]} ……\nFail To Load Key num: {len(no_load_key)}")

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256
    #
    # LOGGER.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # LOGGER.info("actual lr: %.2e" % args.lr)

    LOGGER.info("accumulate grad iterations: %d" % args.accum_iter)
    LOGGER.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # 多卡分布式打开
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        # 单机单卡测试分布式
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], find_unused_parameters=False)

        model_without_ddp = model.module

    # Optimizers
    # 根据batch_size 自适应调整学习率
    init_lr = args.lr
    min_lr = init_lr * 0.01
    lr_limit_max = 1e-3 if args.optimizer in ['Adam', 'AdamW'] else 5e-2
    lr_limit_min = 1e-4 if args.optimizer in ['Adam', 'AdamW'] else 5e-4
    # 基准 batch_size
    reference_batch_size = 32
    scaling_factor = eff_batch_size / reference_batch_size
    # 动态调整学习率
    init_lr_fit = scaling_factor * min(max(init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = scaling_factor * min(max(min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    # 根据optimizer_type 选择优化器
    optimizer = {
        'Adam': torch.optim.Adam(param_groups, lr=init_lr_fit, betas=(args.momentum, 0.999),
                                 weight_decay=args.weight_decay),
        'AdamW': torch.optim.AdamW(param_groups, lr=init_lr_fit, betas=(args.momentum, 0.999),
                                   weight_decay=args.weight_decay),
        'SGD': torch.optim.SGD(param_groups, lr=init_lr_fit, momentum=args.momentum, nesterov=True,
                               weight_decay=args.weight_decay)
    }[args.optimizer]
    # 获得学习率下降公式
    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, init_lr_fit, min_lr_fit, args.epochs)

    loss_scaler = NativeScaler()
    # resume
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    best_acc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        epoch = epoch + 1
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        # train
        train_loss, train_acc, lr = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            loss_scaler=loss_scaler,
            device=device,
            epoch=epoch,
        )
        if log_writer is not None:
            log_writer.append_train_value(epoch, train_loss, train_acc, lr)

        # validate
        if misc.is_main_process() and epoch % args.eval_period == 0:
            test_loss, test_acc = evaluate(
                model=model_without_ddp,
                data_loader=test_dataloader,
                device=device,
                epoch=epoch
            )

            if log_writer is not None:
                log_writer.append_val_value(epoch, test_loss, test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                save_file = os.path.join(args.save_path, 'best_model.pth')
                torch.save(model_without_ddp.state_dict(), save_file)
                LOGGER.info(f"✅ Best model saved to {save_file} with accuracy: {best_acc:.4f}")

        # save checkpoint
        if misc.is_main_process() and (epoch % args.save_period == 0):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler=loss_scaler, epoch=epoch)
            save_file = os.path.join(args.save_path, f'checkpoint-epoch-{epoch}.pth')
            LOGGER.info(f"✅ Checkpoint saved to {save_file} at epoch {epoch}")
            
        # if misc.is_main_process() and epoch % args.save_period == 0:
        #     save_file = os.path.join(args.save_path, f'checkpoint_epoch_{epoch}.pth')
        #     torch.save(model_without_ddp.state_dict(), save_file)
        #     LOGGER.info(f"✅ Checkpoint saved to {save_file} at epoch {epoch}")

    total_time = time.time() - start_time
    if misc.is_main_process():
        LOGGER.info('Training complete in {:.2f} hours.'.format(total_time / 3600))







