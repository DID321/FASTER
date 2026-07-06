"""
@time: 2026/02/11
@file: callbacks.py
@author: WD                     ___       __   ________
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ /
                                ____/|__/      /_____/

"""
import os
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import scipy.signal


class LogWriter:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.lr = []
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag, img_tensor, global_step=None, dataformats='HWC'):
        self.writer.add_image(tag, img_tensor, global_step, dataformats=dataformats)

    def append_train_value(self, epoch, iter_step, total_step, loss_dict, lr):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.train_loss.append(loss_dict["loss"])
        self.lr.append(lr)

        with open(os.path.join(self.log_dir, "epoch_log_train.txt"), 'a') as f:
            f.write("[Train Epoch: {} | {}/{}] Loss: {}  MAE_Loss: {}  SupCL_Loss: {}  SupCL_Cls_Loss: {}  SupCL_Angle_Loss: {}  SupCL_Pitch_Loss: {} SupCL_Polar_Loss: {} SupCL_Scene_Loss: {} Lr: {}\n"
                    .format(str(epoch), str(iter_step), str(total_step), str(loss_dict["loss"]), str(loss_dict["mae_loss"]), str(loss_dict["supcl_loss"]),
                            str(loss_dict["supcl_cls_loss"]), str(loss_dict["supcl_angle_loss"]), str(loss_dict["supcl_pitch_loss"]), str(loss_dict["supcl_polar_loss"]), str(loss_dict["supcl_scene_loss"]), str(lr)))
        # self.writer.add_scalar('Lr', lr, epoch)
        # self.writer.add_scalar('Train/Train_Loss', train_loss, epoch)
        # self.writer.add_scalar('Train/Train_Acc', train_acc, epoch)
        # self.plot_loss()
        # self.plot_mae()
        # self.plot_f1()

    def append_supcl_train_value(self, epoch, iter_step, total_step, loss_dict, lr):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.train_loss.append(loss_dict["loss"])
        self.lr.append(lr)

        with open(os.path.join(self.log_dir, "epoch_log_train.txt"), 'a') as f:
            f.write("[Train Epoch: {} | {}/{}] Loss: {} SupCL_Cls_Loss: {}  SupCL_Angle_Loss: {}  SupCL_Pitch_Loss: {} SupCL_Polar_Loss: {} SupCL_Scene_Loss: {} Lr: {}\n"
                    .format(str(epoch), str(iter_step), str(total_step), str(loss_dict["loss"]),
                            str(loss_dict["supcl_cls_loss"]), str(loss_dict["supcl_angle_loss"]), str(loss_dict["supcl_pitch_loss"]), str(loss_dict["supcl_polar_loss"]), str(loss_dict["supcl_scene_loss"]), str(lr)))
        # self.writer.add_scalar('Lr', lr, epoch)
        # self.writer.add_scalar('Train/Train_Loss', train_loss, epoch)
        # self.writer.add_scalar('Train/Train_Acc', train_acc, epoch)
        # self.plot_loss()
        # self.plot_mae()
        # self.plot_f1()

    def append_val_value(self, epoch, val_loss, val_acc):
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

        with open(os.path.join(self.log_dir, "epoch_log_val.txt"), 'a') as f:
            f.write("{} {} {}\n".format(str(epoch), str(val_loss), str(val_acc)))
        self.writer.add_scalar('Val/Val_Loss', val_loss, epoch)
        self.writer.add_scalar('Val/Val_Acc', val_acc, epoch)
        self.plot_loss()
        self.plot_acc()

    def plot_loss(self):
        train_iters = range(len(self.train_loss))
        val_iters = range(len(self.val_loss))
        plt.figure()
        plt.plot(train_iters, self.train_loss, 'red', linewidth=2, label='train loss')
        plt.plot(val_iters, self.val_loss, 'blue', linewidth=2, label='val loss')

        try:
            if len(self.train_loss) < 25:
                num = 5
            else:
                num = 15

            plt.plot(train_iters, scipy.signal.savgol_filter(self.train_loss, num, 3), 'green', linestyle='--',
                     linewidth=2,
                     label='smooth train loss')
            plt.plot(val_iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                     linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")

    def plot_acc(self):
        train_iters = range(len(self.train_acc))
        val_iters = range(len(self.val_acc))
        plt.figure()
        plt.plot(train_iters, self.train_acc, 'red', linewidth=2, label='train Acc')
        plt.plot(val_iters, self.val_acc, 'blue', linewidth=2, label='val Acc')

        try:
            if len(self.train_acc) < 25:
                num = 5
            else:
                num = 15

            plt.plot(train_iters, scipy.signal.savgol_filter(self.train_acc, num, 3), 'green', linestyle='--',
                     linewidth=2,
                     label='smooth train Acc')
            plt.plot(val_iters, scipy.signal.savgol_filter(self.val_acc, num, 3), '#8B4513', linestyle='--',
                     linewidth=2,
                     label='smooth val Acc')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_Acc.png"))
        plt.cla()
        plt.close("all")

    def plot_f1(self):
        train_iters = range(len(self.train_f1))
        val_iters = range(len(self.val_f1))
        plt.figure()
        plt.plot(train_iters, self.train_f1, 'red', linewidth=2, label='train F1')
        plt.plot(val_iters, self.val_f1, 'blue', linewidth=2, label='val F1')

        try:
            if len(self.train_f1) < 25:
                num = 5
            else:
                num = 15

            plt.plot(train_iters, scipy.signal.savgol_filter(self.train_f1, num, 3), 'green', linestyle='--',
                     linewidth=2,
                     label='smooth train F1')
            plt.plot(val_iters, scipy.signal.savgol_filter(self.val_f1, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val F1')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_F1.png"))
        plt.cla()
        plt.close("all")
