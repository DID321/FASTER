import math
import sys
import torch
from typing import Iterable
import Pretrain.utils.misc as misc
import Pretrain.utils.lr_sched as lr_sched
from Pretrain.utils import LOGGER
from Pretrain.utils.losses import SupConLoss, SupConWeightLoss, get_mask, get_pitch_weights, get_angle_weights
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    total_loss = torch.tensor(0.0, device=device)
    total_supcl_cls_loss = torch.tensor(0.0, device=device)
    total_supcl_angle_loss = torch.tensor(0.0, device=device)
    total_supcl_pitch_loss = torch.tensor(0.0, device=device)
    total_supcl_polar_loss = torch.tensor(0.0, device=device)
    total_supcl_scene_loss = torch.tensor(0.0, device=device)
    lr = torch.tensor(0.0, device=device)
    # metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    accum_iter = args.accum_iter

    supcon_criterion = SupConLoss(temperature=0.1)
    supcon_weight_criterion = SupConWeightLoss(temperature=0.1)

    optimizer.zero_grad()

    if misc.is_main_process():
        data_loader_par = tqdm(data_loader, total=len(data_loader))
        # data_loader_par = tqdm(metric_logger.log_every(data_loader, print_freq, header), total=len(data_loader))
    else:
        data_loader_par = data_loader
        # data_loader_par = metric_logger.log_every(data_loader, print_freq, header)

    img, pred, mask = None, None, None
    batch_num = len(data_loader)
    for data_iter_step, data in enumerate(data_loader_par):
        # sup contrastive learning data preparation
        supcl_images = torch.cat([data['img'], data['img_neg_cls'], data['img_pos_angle'], data['img_pos_pitch'], data['img_pos_polar'], data['img_pos_scene']], dim=0)
        # supcl_images = torch.cat([data['img'], data['img_neg_cls'], data['img_pos_angle'], data['img_pos_pitch']], dim=0)
        cls_labels, angle_labels, pitch_labels, polar_labels, scene_labels = data['cls'], data['angle'], data['pitch'], data['polar'], data['scene']
        neg_cls, pos_angles, pos_pitchs, pos_polars, pos_scenes = data['neg_cls'], data['pos_angle'], data['pos_pitch'], data['pos_polar'], data['pos_scene']

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.no_grad():
            supcl_images = supcl_images.to(device, non_blocking=True)
            cls_labels, angle_labels, pitch_labels, polar_labels, scene_labels = cls_labels.to(device, non_blocking=True), angle_labels.to(device, non_blocking=True), pitch_labels.to(device, non_blocking=True), polar_labels.to(device, non_blocking=True), scene_labels.to(device, non_blocking=True)
            neg_cls, pos_angles, pos_pitchs, pos_polars, pos_scenes = neg_cls.to(device, non_blocking=True), pos_angles.to(device, non_blocking=True), pos_pitchs.to(device, non_blocking=True), pos_polars.to(device, non_blocking=True), pos_scenes.to(device, non_blocking=True)
        bsz = cls_labels.shape[0]

        # with torch.cuda.amp.autocast():
        # sup contrastive learning forward
        embed = model(supcl_images)
        embed_img, embed_neg_cls, embed_pos_angle, embed_pos_pitch, embed_pos_polar, embed_pos_scene = torch.split(embed, [bsz, bsz, bsz, bsz, bsz, bsz], dim=0)
        # embed = torch.cat([embed_img.unsqueeze(1), embed_neg_cls.unsqueeze(1), embed_pos_angle.unsqueeze(1), embed_pos_pitch.unsqueeze(1), embed_pos_polar.unsqueeze(1), embed_pos_scene.unsqueeze(1)], dim=1)
        # embed_img, embed_neg_cls, embed_pos_angle, embed_pos_pitch = torch.split(embed, [bsz, bsz, bsz, bsz], dim=0)
        # embed = torch.cat([embed_img.unsqueeze(1), embed_neg_cls.unsqueeze(1), embed_pos_angle.unsqueeze(1), embed_pos_pitch.unsqueeze(1)], dim=1)
        # 不同层次正负样本对比学习
        embed_img_neg_cls = torch.cat([embed_img.unsqueeze(1), embed_neg_cls.unsqueeze(1)], dim=1)
        embed_img_pos_angle = torch.cat([embed_img.unsqueeze(1), embed_pos_angle.unsqueeze(1)], dim=1)
        embed_img_pos_pitch = torch.cat([embed_img.unsqueeze(1), embed_pos_pitch.unsqueeze(1)], dim=1)
        embed_img_pos_polar = torch.cat([embed_img.unsqueeze(1), embed_pos_polar.unsqueeze(1)], dim=1)
        embed_img_pos_scene = torch.cat([embed_img.unsqueeze(1), embed_pos_scene.unsqueeze(1)], dim=1)

        # sup contrastive learning loss
        mask_cls = get_mask(cls_labels, neg_cls)
        supcl_cls_loss = supcon_criterion(embed_img_neg_cls, mask=mask_cls)

        angle_weights = get_angle_weights(angle_labels, pos_angles)
        supcl_angle_loss = supcon_weight_criterion(embed_img_pos_angle, labels=cls_labels, weights=angle_weights)

        pitch_weights = get_pitch_weights(pitch_labels, pos_pitchs)
        supcl_pitch_loss = supcon_weight_criterion(embed_img_pos_pitch, labels=cls_labels, weights=pitch_weights)

        supcl_polar_loss = supcon_criterion(embed_img_pos_polar, labels=cls_labels)
        supcl_scene_loss = supcon_criterion(embed_img_pos_scene, labels=cls_labels)

        # supcl_loss = supcl_cls_loss + supcl_angle_loss + supcl_pitch_loss
        loss = 0.2 * supcl_cls_loss + 0.2 * supcl_angle_loss + 0.2 * supcl_pitch_loss + 0.2 * supcl_polar_loss + 0.2 * supcl_scene_loss

        total_loss += loss
        total_supcl_cls_loss += supcl_cls_loss
        total_supcl_angle_loss += supcl_angle_loss
        total_supcl_pitch_loss += supcl_pitch_loss
        total_supcl_polar_loss += supcl_polar_loss
        total_supcl_scene_loss += supcl_scene_loss

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        loss /= accum_iter

        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print('The None grad model is :', name)

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        # metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        supcl_cls_loss_reduce = misc.all_reduce_mean(supcl_cls_loss.item())
        supcl_angle_loss_reduce = misc.all_reduce_mean(supcl_angle_loss.item())
        supcl_pitch_loss_reduce = misc.all_reduce_mean(supcl_pitch_loss.item())
        supcl_polar_loss_reduce = misc.all_reduce_mean(supcl_polar_loss.item())
        supcl_scene_loss_reduce = misc.all_reduce_mean(supcl_scene_loss.item())

        loss_dict = {
            "loss": loss_value_reduce,
            "supcl_cls_loss": supcl_cls_loss_reduce,
            "supcl_angle_loss": supcl_angle_loss_reduce,
            "supcl_pitch_loss": supcl_pitch_loss_reduce,
            "supcl_polar_loss": supcl_polar_loss_reduce,
            "supcl_scene_loss": supcl_scene_loss_reduce
        }

        if misc.is_main_process():
            # data_loader_par.desc = "[train epoch {}] loss: {:.4f}, lr: {:.6f}".format(epoch, loss_value_reduce, lr)
            # if (data_iter_step + 1) % print_freq == 0 and log_writer is not None:
            #     log_writer.append_train_value(epoch, data_iter_step + 1, batch_num, loss_value_reduce, lr)

            # data_loader_par.desc = "[train epoch {}] loss: {:.4f}, mae_loss: {:.4f}, supcl_loss: {:.4f}, supcl_cls_loss: {:.4f}, supcl_angle_loss: {:.4f}, supcl_pitch_loss: {:.4f}, lr: {:.6f}".format(
            #     epoch, loss_value_reduce, mae_loss_value_reduce, supcl_loss_value_reduce, supcl_cls_loss_reduce, supcl_angle_loss_reduce, supcl_pitch_loss_reduce, lr
            # )
            data_loader_par.desc = "[train epoch {}] loss: {:.4f}, supcl_cls_loss: {:.4f}, supcl_angle_loss: {:.4f}, supcl_pitch_loss: {:.4f}, supcl_polar_loss: {:.4f}, supcl_scene_loss: {:.4f}, lr: {:.6f}".format(
                epoch, loss_value_reduce, supcl_cls_loss_reduce,
                supcl_angle_loss_reduce, supcl_pitch_loss_reduce, supcl_polar_loss_reduce, supcl_scene_loss_reduce, lr
            )
            if (data_iter_step + 1) % print_freq == 0 and log_writer is not None:
                log_writer.append_supcl_train_value(epoch, data_iter_step + 1, batch_num, loss_dict, lr)

        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        # if misc.is_main_process() and log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        #     log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
        #     log_writer.add_scalar('lr', lr, epoch_1000x)

    # Synchronize metrics across GPUs
    total_loss = misc.all_reduce(total_loss)
    total_supcl_cls_loss = misc.all_reduce(total_supcl_cls_loss)
    total_supcl_angle_loss = misc.all_reduce(total_supcl_angle_loss)
    total_supcl_pitch_loss = misc.all_reduce(total_supcl_pitch_loss)
    total_supcl_polar_loss = misc.all_reduce(total_supcl_polar_loss)
    total_supcl_scene_loss = misc.all_reduce(total_supcl_scene_loss)

    batch_num = misc.all_reduce(batch_num)

    avg_loss = total_loss / batch_num
    avg_supcl_cls_loss = total_supcl_cls_loss / batch_num
    avg_supcl_angle_loss = total_supcl_angle_loss / batch_num
    avg_supcl_pitch_loss = total_supcl_pitch_loss / batch_num
    avg_supcl_polar_loss = total_supcl_polar_loss / batch_num
    avg_supcl_scene_loss = total_supcl_scene_loss / batch_num

    if misc.is_main_process() and log_writer is not None:
        log_writer.add_scalar('train_loss', avg_loss, epoch)
        log_writer.add_scalar('train_supcl_loss_cls', avg_supcl_cls_loss, epoch)
        log_writer.add_scalar('train_supcl_loss_angle', avg_supcl_angle_loss, epoch)
        log_writer.add_scalar('train_supcl_loss_pitch', avg_supcl_pitch_loss, epoch)
        log_writer.add_scalar('train_supcl_loss_polar', avg_supcl_polar_loss, epoch)
        log_writer.add_scalar('train_supcl_loss_scene', avg_supcl_scene_loss, epoch)

        log_writer.add_scalar('lr', lr, epoch)
        LOGGER.info("Epoch {}: Average Loss: {:.4f}, lr: {:.6f}".format(epoch, avg_loss, lr))

    # if misc.is_main_process() and (epoch % 10 == 0 or epoch == args.epochs):
    # # if misc.is_main_process() or epoch == args.epochs:
    #     # visualize the mae results
    #     img, masked_img, pred_img0,  pred_img1, pred_img2 = model.visualize_mae_results(img=img, pred=pred,mask=mask)
    #     log_writer.add_image('MAE/images', img, epoch, dataformats='HWC')
    #     log_writer.add_image('MAE/masked_images', masked_img, epoch, dataformats='HWC')
    #     log_writer.add_image('MAE/pred_images0', pred_img0, epoch, dataformats='HW')
    #     log_writer.add_image('MAE/pred_images1', pred_img1, epoch, dataformats='HW')
    #     log_writer.add_image('MAE/pred_images2', pred_img2, epoch, dataformats='HW')

    return avg_loss, lr
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}