import logging
import subprocess
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='train.log')


def write_log(cmd, status):
    if not status:
        logging.info(f'{cmd} train success')
    else:
        logging.info(f'{cmd} train fail')

cmds =[
    # ['SARATRX', './logs_100/SARATRX_EOC_angle', './logs_100/SARATRX_EOC_angle', './cfg/datasets/EOC_angle.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_EOC_angle_torch_dist.txt', './logs_100/SARATRX_EOC_angle/last_model.pth'],
    # ['SARATRX', './logs_100/SARATRX_EOC_pitch', './logs_100/SARATRX_EOC_pitch', './cfg/datasets/EOC_pitch.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_EOC_pitch_torch_dist.txt', './logs_100/SARATRX_EOC_pitch/last_model.pth'],
    # ['SARATRX', './logs_100/SARATRX_EOC_polar', './logs_100/SARATRX_EOC_polar', './cfg/datasets/EOC_polar.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_EOC_polar_torch_dist.txt'],
    # ['SARATRX', './logs_100/SARATRX_EOC_scene', './logs_100/SARATRX_EOC_scene', './cfg/datasets/EOC_scene.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_EOC_scene_torch_dist.txt', './logs_100/SARATRX_EOC_scene/last_model.pth'],
    # ['SARATRX', './logs_100/SARATRX_EOC_scene_Grass2Road', './logs_100/SARATRX_EOC_scene_Grass2Road', './cfg/datasets/EOC_scene_Grass2Road.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_EOC_scene_Grass2Road_torch_dist.txt', './logs_100/SARATRX_EOC_scene_Grass2Road/checkpoint-epoch-25.pth'],

    # ['ViT', './logs_100/ViT_EOC_angle', './logs_100/ViT_EOC_angle', './cfg/datasets/EOC_angle.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ViT_EOC_angle_torch_dist.txt', ''],
    # ['ViT', './logs_100/ViT_EOC_pitch', './logs_100/ViT_EOC_pitch', './cfg/datasets/EOC_pitch.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ViT_EOC_pitch_torch_dist.txt'],
    # ['ViT', './logs_100/ViT_EOC_polar', './logs_100/ViT_EOC_polar', './cfg/datasets/EOC_polar.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ViT_EOC_polar_torch_dist.txt', './logs_100/ViT_EOC_polar/last_model.pth'],
    # ['ViT', './logs_100/ViT_EOC_scene', './logs_100/ViT_EOC_scene', './cfg/datasets/EOC_scene.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ViT_EOC_scene_torch_dist.txt', './logs_100/ViT_EOC_scene/last_model.pth'],
    # ['ViT', './logs_100/ViT_EOC_scene_Grass2Road', './logs_100/ViT_EOC_scene_Grass2Road', './cfg/datasets/EOC_scene_Grass2Road.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ViT_EOC_scene_Grass2Road_torch_dist.txt', './logs_100/ViT_EOC_scene_Grass2Road/last_model.pth'],

    # ['HiViT_SupCL', './logs_100/HiViT_SupCL_100_EOC_angle', './logs_100/HiViT_SupCL_100_EOC_angle', './cfg/datasets/EOC_angle.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/HiViT_SupCL_100_EOC_angle_torch_dist.txt'],
    # ['HiViT_SupCL', './logs_100/HiViT_SupCL_100_EOC_pitch', './logs_100/HiViT_SupCL_100_EOC_pitch', './cfg/datasets/EOC_pitch.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/HiViT_SupCL_100_EOC_pitch_torch_dist.txt'],
     # ConvNeXt
    # ['ConvNeXt', './logs_100/ConvNeXt_EOC_angle', './logs_100/ConvNeXt_EOC_angle', './cfg/datasets/EOC_angle.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_angle_torch_dist.txt'],
    # ['ConvNeXt', './logs_100/ConvNeXt_EOC_pitch', './logs_100/ConvNeXt_EOC_pitch', './cfg/datasets/EOC_pitch.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_pitch_torch_dist.txt'],
    # ['ConvNeXt', './logs_100/ConvNeXt_EOC_polar', './logs_100/ConvNeXt_EOC_polar', './cfg/datasets/EOC_polar.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_polar_torch_dist.txt', './logs_100/ConvNeXt_EOC_polar/last_model.pth'],
    # ['ConvNeXt', './logs_100/ConvNeXt_EOC_scene', './logs_100/ConvNeXt_EOC_scene', './cfg/datasets/EOC_scene.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_scene_torch_dist.txt'],
    # ['ConvNeXt', './logs_100/ConvNeXt_EOC_scene_Grass2Road', './logs_100/ConvNeXt_EOC_scene_Grass2Road', './cfg/datasets/EOC_scene_Grass2Road.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_scene_Grass2Road_torch_dist.txt'],

    # ['ConvNeXt', './logs_100/ConvNeXt_pretrain_EOC_angle', './logs_100/ConvNeXt_pretrain_EOC_angle', './cfg/datasets/EOC_angle.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_angle_torch_dist.txt'],
    # ['ConvNeXt', './logs_100/ConvNeXt_pretrain_EOC_pitch', './logs_100/ConvNeXt_pretrain_EOC_pitch', './cfg/datasets/EOC_pitch.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_pitch_torch_dist.txt', './logs_100/ConvNeXt_pretrain_EOC_pitch/last_model.pth'],
    # ['ConvNeXt', './logs_100/ConvNeXt_pretrain_EOC_polar', './logs_100/ConvNeXt_pretrain_EOC_polar', './cfg/datasets/EOC_polar.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_polar_torch_dist.txt'],
    # ['ConvNeXt', './logs_100/ConvNeXt_pretrain_EOC_scene', './logs_100/ConvNeXt_pretrain_EOC_scene', './cfg/datasets/EOC_scene.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_scene_torch_dist.txt', './logs_100/ConvNeXt_pretrain_EOC_scene/last_model.pth'],
    # ['ConvNeXt', './logs_100/ConvNeXt_pretrain_EOC_scene_Grass2Road', './logs_100/ConvNeXt_pretrain_EOC_scene_Grass2Road', './cfg/datasets/EOC_scene_Grass2Road.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_scene_Grass2Road_torch_dist.txt', './logs_100/ConvNeXt_pretrain_EOC_scene_Grass2Road/last_model.pth'],
    # ['ConvNeXt', './logs_fewtrain_azimuth/ConvNeXt_EOC_depression_20', './logs_fewtrain_azimuth/ConvNeXt_EOC_depression_20', './cfg/datasets/EOC_fewtrain_depression_20.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_fewtrain_depression_20_torch_dist.txt'],
    # ['ConvNeXt', './logs_fewtrain_azimuth/ConvNeXt_EOC_depression_40', './logs_fewtrain_azimuth/ConvNeXt_EOC_depression_40', './cfg/datasets/EOC_fewtrain_depression_40.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_EOC_fewtrain_depression_40_torch_dist.txt'],

    # ResNet18
    # ['ResNet18', './logs_100/ResNet18_EOC_angle', './logs_100/ResNet18_EOC_angle', './cfg/datasets/EOC_angle.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ResNet18_EOC_angle_torch_dist.txt'],
    # ['ResNet18', './logs_100/ResNet18_EOC_pitch', './logs_100/ResNet18_EOC_pitch', './cfg/datasets/EOC_pitch.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ResNet18_EOC_pitch_torch_dist.txt'],
    # ['ResNet18', './logs_100/ResNet18_EOC_polar', './logs_100/ResNet18_EOC_polar', './cf g/datasets/EOC_polar.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ResNet18_EOC_polar_torch_dist.txt', './logs_100/ResNet18_EOC_polar/last_model.pth'],
    # ['ResNet18', './logs_100/ResNet18_EOC_scene', './logs_100/ResNet18_EOC_scene', './cfg/datasets/EOC_scene.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ResNet18_EOC_scene_torch_dist.txt', './logs_100/ResNet18_EOC_scene/last_model.pth'],
    # ['ResNet18', './logs_100/ResNet18_EOC_scene_Grass2Road', './logs_100/ResNet18_EOC_scene_Grass2Road', './cfg/datasets/EOC_scene_Grass2Road.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ResNet18_EOC_scene_Grass2Road_torch_dist.txt'],

    # ['SARATRX', './logs_100/SARATRX_SOC', './logs_100/SARATRX_SOC', './cfg/datasets/SOC.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_SOC_torch_dist.txt'],
    # ['ViT', './logs_100/ViT_SOC', './logs_100/ViT_SOC', './cfg/datasets/SOC.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ViT_SOC_torch_dist.txt', './logs_100/ViT_SOC/last_model.pth'],
    # ['ConvNeXt', './logs_100/ConvNeXt_SOC', './logs_100/ConvNeXt_SOC', './cfg/datasets/SOC.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ConvNeXt_SOC_torch_dist.txt'],
    # ['ResNet18', './logs_100/ResNet18_SOC', './logs_100/ResNet18_SOC', './cfg/datasets/SOC.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/ResNet18_SOC_torch_dist.txt'],

    # few_shot 5
    # ['SARATRX', './logs_100/SARATRX_SOC_fewtrain_5', './logs_100/SARATRX_SOC_fewtrain_5', './cfg/datasets/SOC_fewtrain_5.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_SOC_fewtrain_5_torch_dist.txt'],
    # ['SARATRX', './logs_100/SARATRX_SOC_fewtrain_10', './logs_100/SARATRX_SOC_fewtrain_10', './cfg/datasets/SOC_fewtrain_10.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_SOC_fewtrain_10_torch_dist.txt'],
    # ['SARATRX', './logs_100/SARATRX_SOC_fewtrain_20', './logs_100/SARATRX_SOC_fewtrain_20', './cfg/datasets/SOC_fewtrain_20.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_SOC_fewtrain_20_torch_dist.txt'],
    # ['SARATRX', './logs_100/SARATRX_SOC_fewtrain_40', './logs_100/SARATRX_SOC_fewtrain_40', './cfg/datasets/SOC_fewtrain_40.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_SOC_fewtrain_40_torch_dist.txt'],
    # ['SARATRX', './logs_100/SARATRX_SOC_fewtrain_60', './logs_100/SARATRX_SOC_fewtrain_60', './cfg/datasets/SOC_fewtrain_60.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_SOC_fewtrain_60_torch_dist.txt'],
    # ['SARATRX', './logs_100/SARATRX_SOC_fewtrain_80', './logs_100/SARATRX_SOC_fewtrain_80', './cfg/datasets/SOC_fewtrain_80.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/SARATRX_SOC_fewtrain_80_torch_dist.txt'],
]
def main():
    for cmd in cmds:
        if cmd[0] == 'ViT':
            if len(cmd) == 6:
                res = f"python train_SOC_multi_gpu_file.py --weights ./weights/vit_base_patch16_224.pth --model-name {cmd[0]} --log-path {cmd[1]} --save-path {cmd[2]} --data-cfg {cmd[3]} --dist-url {cmd[4]} --resume {cmd[5]}"
            else:
                res = f"python train_SOC_multi_gpu_file.py --weights ./weights/vit_base_patch16_224.pth --model-name {cmd[0]} --log-path {cmd[1]} --save-path {cmd[2]} --data-cfg {cmd[3]} --dist-url {cmd[4]}"
        elif cmd[0] == 'ConvNeXt':
            if len(cmd) == 6:
                res = f"python train_SOC_multi_gpu_file.py --weights ./weights/convnext_base-6075fbad.pth --model-name {cmd[0]} --log-path {cmd[1]} --save-path {cmd[2]} --data-cfg {cmd[3]} --dist-url {cmd[4]} --resume {cmd[5]}"
            else:
                res = f"python train_SOC_multi_gpu_file.py --weights ./weights/convnext_base-6075fbad.pth --model-name {cmd[0]} --log-path {cmd[1]} --save-path {cmd[2]} --data-cfg {cmd[3]} --dist-url {cmd[4]}"

        elif len(cmd) == 6:
            res = f"python train_SOC_multi_gpu_file.py --model-name {cmd[0]} --log-path {cmd[1]} --save-path {cmd[2]} --data-cfg {cmd[3]} --dist-url {cmd[4]} --resume {cmd[5]}"
        else:
            res = f"python train_SOC_multi_gpu_file.py --model-name {cmd[0]} --log-path {cmd[1]} --save-path {cmd[2]} --data-cfg {cmd[3]} --dist-url {cmd[4]}"

        p = subprocess.Popen(res, shell=True)
        return_code = p.wait()
        write_log(cmd, return_code)


if __name__ == '__main__':
    main()