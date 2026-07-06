"""
@time: 2026/03/10
@file: test_SOC.py
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
from utils.test_engine import model_test
from utils.confusion_matrix import cal_confusion_matrix, plot_confusion_matrix, show_save_acc


def parameter_setting():
    parser = argparse.ArgumentParser()
    # 测试batch_size
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # log 文件夹
    parser.add_argument('--log-path', type=str, default='./logs_100/VGG16_EOC_polar/')
    # 训练图像尺寸
    parser.add_argument('--img-size', type=list, default=[224, 224])
    # 数据集配置文件所在根目录
    parser.add_argument('--data-cfg', type=str, default="./cfg/datasets/EOC_polar_HV.yaml")
    # num_workers
    parser.add_argument('--num-workers', type=int, default=4)
    # 模型
    parser.add_argument('--model-name', type=str, default='VGG16', help='model name')
    # 训练权重路径
    parser.add_argument('--weights', type=str, default='./logs_100/VGG16_EOC_polar/best_model.pth', help='initial weights path')
    # 显卡选择
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
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

    cfg = yaml_load(args.data_cfg)
    # 记得修改cfg["test_txt_path"]
    test_dataset = SARDataSet(data_root=cfg["data_path"], img_txt_path=cfg["test_txt_path"], classes=cfg["names"], sar_config=cfg['sar_test'])

    if global_rank == 0 and args.log_path is not None:
        os.makedirs(args.log_path, exist_ok=True)

    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    LOGGER.info("Sampler_test = %s" % str(test_sampler))

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                                                  num_workers=args.num_workers, collate_fn=SARDataSet.collate_fn)

    LOGGER.info('test shape:{}'.format(len(test_dataloader.dataset)))
    num_classes = len(cfg["names"].keys())
    # Initialize model
    if args.model_name == 'ResNet34':
        model = ResNet_34(num_classes=len(cfg["names"].keys()))
    elif args.model_name == 'ResNet18':
        model = ResNet_18(num_classes=len(cfg["names"].keys()))
    elif args.model_name == 'VGG16':
        model = VGGNet(num_classes=len(cfg["names"].keys()))
    elif args.model_name == 'HiViT':
        model = hivit_base(num_classes=len(cfg["names"].keys()))
    elif args.model_name == 'SARATRX':
        model = saratr_x(num_classes=len(cfg["names"].keys()))
    elif args.model_name == 'ViT':
        model = vit_base_patch16(num_classes=len(cfg["names"].keys()))
    elif args.model_name == 'ConvNeXt':
        model = ConvNeXt_base(num_classes=len(cfg["names"].keys()))
    # model = ResNet_18(num_classes=len(cfg["names"].keys()))
    # model = ConvNeXt_base(num_classes=len(cfg["names"].keys()))
    # model = VGGNet(num_classes=len(cfg["names"].keys()))
    # model = HDANet(num_classes=len(cfg["names"].keys()), in_ch=3)
    # model = vit_base_patch16(num_classes=len(cfg["names"].keys()))
    # model = hivit_base(num_classes=len(cfg["names"].keys()))
    # model = saratr_x(num_classes=len(cfg["names"].keys()))

    # model = ResNet_50(num_classes=len(cfg["names"].keys()))
    # summary(model, input_data=[torch.randn((args.batch_size, 3, args.img_size[0], args.img_size[1]))], device=device)
    if args.weights != '':
        assert os.path.exists(args.weights), "⚠️ Warning! Weights file: '{}' not exist.".format(args.weights)
        model_dict = model.state_dict()
        if args.model_name == 'SARATRX':
            checkpoint = torch.load(args.weights, map_location='cpu')['model']
        else:
            checkpoint = torch.load(args.weights, map_location='cpu')
        # if args.model_name == 'SARATRX':
        #     checkpoint = torch.load(args.weights, map_location='cpu')['model']
        # else:
        #     checkpoint = torch.load(args.weights, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in checkpoint.items():
            # k = 'model.' + k
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

    eff_batch_size = args.batch_size * misc.get_world_size()

    LOGGER.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # 多卡分布式打开
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        # 单机单卡测试分布式
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], find_unused_parameters=False)

        model_without_ddp = model.module

    print(f"Start testing for {cfg['task']} ...")
    start_time = time.time()

    preds, acc1, acc5 = model_test(
        model=model_without_ddp,
        data_loader=test_dataloader,
        device=device,
    )
    ground_truth = np.array([lb['cls'] for lb in test_dataset.labels])
    cm_norm = cal_confusion_matrix(preds, ground_truth, num_classes, is_norm=True)
    cm = cal_confusion_matrix(preds, ground_truth, num_classes, is_norm=False)

    save_path_cm = pathlib.Path(args.log_path) / ('confusion_matrix_' + cfg['task'] + '.png')
    save_path_result = pathlib.Path(args.log_path) / ('test_results_' + cfg['task'] + '.txt')

    plot_confusion_matrix(cm, cfg["names"].values(), save_path=save_path_cm)
    LOGGER.info('✅ plot confusion matrix in {}!'.format(save_path_cm))

    show_save_acc(acc1, acc5, cm_norm, cfg["names"].values(), save_path=save_path_result)
    LOGGER.info('✅ save test results in {}!'.format(save_path_result))
    total_time = time.time() - start_time
    LOGGER.info('Test complete in {:.4f} s.'.format(total_time))
