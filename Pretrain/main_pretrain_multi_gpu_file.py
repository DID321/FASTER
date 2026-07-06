"""
@time: 2026/02/19
@file: main_pretrain.py
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
import datetime
import pathlib
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
# 将项目根目录加入 Python 路径
ROOT = pathlib.Path(__file__).parent.parent
# print(ROOT)
sys.path.insert(0, str(ROOT))
import argparse
import utils.misc as misc
import timm.optim.optim_factory as optim_factory
import collections
# from Pretrain.models.hivit_mae import mae_hivit_base_dec512d6b
from Pretrain.models.hivit_mae_scl import supcl_mae_hivit_base_dec512d6b
from Pretrain.models.vit_mae_scl import supcl_mae_vit_base_patch16_dec512d8b
from Pretrain.utils import LOGGER, yaml_load, show_config
from Pretrain.utils.misc import NativeScalerWithGradNormCount as NativeScaler
from Pretrain.utils.callbacks import LogWriter
from Pretrain.utils.engine_pretrain import train_one_epoch
from Pretrain.data.datasets import SupCLSARDataSet
from Pretrain.data.augment import RandomResizedCenterCrop

# from utils.TrainTest import model_train, model_val, model_test
# from model.Model import convnext_1, ResNet_34

def parameter_setting():
    parser = argparse.ArgumentParser('SAR pre-training', add_help=False)
    # 训练迭代次数
    parser.add_argument('--epochs', type=int, default=200)
    # 训练batch_size
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (AdamW, Adam, SGD)')
    # 学习率衰减策略
    parser.add_argument('--lr-decay-type', type=str, default='Cos', help='lr decay type (Step, Cos)')
    # 学习率
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')

    # 动量
    parser.add_argument('--momentum', type=float, default=0.9)
    # 衰减速率
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    # 掩码率
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')
    # 是否使用归一化像素作为损失目标
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    # MAE 损失权重
    parser.add_argument('--mae_loss_weight', type=float, default=0.5, help='weight for mae loss')
    # sup contrastive loss 权重
    parser.add_argument('--supcl_loss_weight', type=float, default=0.5, help='weight for sup contrastive loss')
    # 预热epoch数
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')

    # log 文件夹
    parser.add_argument('--log-path', type=str, default='./logs/ViT_SupCL_MAE_Prtrain_SOC')
    # 保存权重文件夹
    parser.add_argument('--save-path', type=str, default='./logs/ViT_SupCL_MAE_Prtrain_SOC')
    # 训练图像尺寸
    parser.add_argument('--img-size', type=list, default=[224, 224])
    # 数据集配置文件所在根目录
    parser.add_argument('--data-cfg', type=str, default="./cfg/datasets/Pretrain_SOC.yaml")
    # 多少epoch保存一次权重
    parser.add_argument('--save-period', type=int, default=20)
    # 多少epoch验证一次
    parser.add_argument('--eval-period', type=int, default=1)
    # num_workers
    parser.add_argument('--num-workers', type=int, default=4)
    # 预训练权重路径，如果不想载入就设置为空字符 # ./weights/resnet18-f37072fd.pth mae_hivit_base_1600ep
    parser.add_argument('--weights', type=str, default='./weights/vit_base_patch16_224.pth', help='initial weights path')
    # 是否从断点恢复训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 是否冻结head以外所有权重
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    # 随机数种子
    parser.add_argument('--seed', type=int, default=0)
    # 显卡选择
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    
    parser.add_argument('--accum-iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # distributed training parameters
    # 所有参与训练的进程总数 通常等于 GPU 总数
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    # 用于指定该进程应该使用哪块 GPU
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist-on-itp', action='store_true')
    # parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist-url', default='file:///J:/CSAR_ATR_Bench_20260223/Pretrain_ViT_torch_dist_init.txt', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def main(rank, *args):
    # os.environ["RANK"] = str(rank)
    # os.environ["LOCAL_RANK"] = str(rank)

    args = parameter_setting()
    show_config(args.__dict__)
    
    # Setup distributed training
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    device = torch.device(args.device)
    LOGGER.info("✅ Using {} device.".format(device))
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    # torch.cuda.manual_seed(args.seed)

    # load dataset config
    cfg = yaml_load(args.data_cfg)

    # simple augmentation
    transform_train = transforms.Compose([
            RandomResizedCenterCrop(
                target_size=args.img_size,    # 最终输出尺寸
                max_offset=0.2,                 # 中心偏移最大比例
                area_ratio_range=(0.7, 1.0),    # 面积比例范围
                aspect_ratio_range=(1.0, 1.0),  # 宽高比范围
                interpolation=3
            ),
            transforms.ToTensor()
    ])

    # 记得修改cfg["test_txt_path"]
    # train_dataset = SARDataSet(data_root=cfg["data_path"], img_txt_path=cfg["train_txt_path"], classes=cfg["names"], sar_config=cfg['sar_train'], transform=transform_train)

    train_dataset = SupCLSARDataSet(data_root=cfg["data_path"], img_txt_path=cfg["train_txt_path"], classes=cfg["names"], sar_config=cfg['sar_train'], transform=transform_train)
    # train_dataset = SARDataSet(data_root=cfg["data_path"], img_txt_path=cfg["train_txt_path"], classes=cfg["names"],
    #                            sar_config=cfg['sar_train'], transform=None)

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
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    
    LOGGER.info("Sampler_train = %s" % str(train_sampler))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=args.num_workers, collate_fn=SupCLSARDataSet.collate_fn)

    LOGGER.info('train shape: {} '.format(len(train_dataloader.dataset)))

    # Initialize model
    # model = mae_hivit_base_dec512d6b(norm_pix_loss=args.norm_pix_loss)
    # model = supcl_mae_hivit_base_dec512d6b(norm_pix_loss=args.norm_pix_loss)

    model = supcl_mae_vit_base_patch16_dec512d8b(norm_pix_loss=args.norm_pix_loss)
    # model = ResNet_50(num_classes=len(cfg["names"].keys()))
    # summary(model, input_data=[torch.randn((args.batch_size, 3, args.img_size[0], args.img_size[1]))], device=device)
    if args.weights != '':
        assert os.path.exists(args.weights), "⚠️ Warning! Weights file: '{}' not exist.".format(args.weights)
        checkpoint = torch.load(args.weights, map_location='cpu')
        # load pre-trained model
        msg = model.load_state_dict(checkpoint, strict=False)
        LOGGER.info(f"✅ Successful Load Weights: {args.weights}")
        LOGGER.info(msg)
    
    model.to(device)
    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    LOGGER.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    LOGGER.info("actual lr: %.2e" % args.lr)

    LOGGER.info("accumulate grad iterations: %d" % args.accum_iter)
    LOGGER.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # 多卡分布式打开
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        # 单机单卡测试分布式
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], find_unused_parameters=False)

        model_without_ddp = model.module

    # Optimizers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    # 根据optimizer_type 选择优化器
    optimizer = {
        'Adam': torch.optim.Adam(param_groups, lr=args.lr, betas=(args.momentum, 0.95),
                                 weight_decay=args.weight_decay),
        'AdamW': torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.momentum, 0.95),
                                   weight_decay=args.weight_decay),
        'SGD': torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, nesterov=True,
                               weight_decay=args.weight_decay)
    }[args.optimizer]

    loss_scaler = NativeScaler()
    # resume
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    LOGGER.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        epoch = epoch + 1
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        # train
        train_loss, lr = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            log_writer=log_writer,
            args=args
        )
        # save checkpoint
        if misc.is_main_process() and (epoch % args.save_period == 0 or epoch == args.epochs):
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler=loss_scaler, epoch=epoch)
            save_file = os.path.join(args.save_path, f'checkpoint-epoch-{epoch}.pth')
            LOGGER.info(f"✅ Checkpoint saved to {save_file} at epoch {epoch}")
            
        # if misc.is_main_process() and epoch % args.save_period == 0:
        #     save_file = os.path.join(args.save_path, f'checkpoint_epoch_{epoch}.pth')
        #     torch.save(model_without_ddp.state_dict(), save_file)
        #     LOGGER.info(f"✅ Checkpoint saved to {save_file} at epoch {epoch}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        LOGGER.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main(rank=0)
    # 设置 spawn 启动方式（Windows 必须）
    # torch.multiprocessing.set_start_method('spawn', force=True)
    #
    # world_size = 2  # 2 张卡
    #
    # torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)







