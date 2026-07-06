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

    ['HiViT', './logs_100/HiViT_EOC_angle', './logs_100/HiViT_EOC_angle', './cfg/datasets/EOC_angle.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/HiViT_EOC_angle_torch_dist.txt'],
    ['HiViT', './logs_100/HiViT_EOC_pitch', './logs_100/HiViT_EOC_pitch', './cfg/datasets/EOC_pitch.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/HiViT_EOC_pitch_torch_dist.txt'],
    ['HiViT', './logs_100/HiViT_EOC_polar', './logs_100/HiViT_EOC_polar', './cfg/datasets/EOC_polar.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/HiViT_EOC_polar_torch_dist.txt'],
    ['HiViT', './logs_100/HiViT_EOC_scene', './logs_100/HiViT_EOC_scene', './cfg/datasets/EOC_scene.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/HiViT_EOC_scene_torch_dist.txt'],
    ['HiViT', './logs_100/HiViT_EOC_scene_Grass2Road', './logs_100/HiViT_EOC_scene_Grass2Road', './cfg/datasets/EOC_scene_Grass2Road.yaml', 'file:///J:/CSAR_ATR_Bench_20260223/HiViT_EOC_scene_Grass2Road_torch_dist.txt'],

]
def main():
    for cmd in cmds:
        res = f"python train_SOC_multi_gpu_file.py --weights ./weights/mae_hivit_base_1600ep.pth --model-name {cmd[0]} --log-path {cmd[1]} --save-path {cmd[2]} --data-cfg {cmd[3]} --dist-url {cmd[4]}"

        # res = f"python train_SOC_multi_gpu_file_100.py --model-name {cmd[0]} --log-path {cmd[1]} --save-path {cmd[2]} --data-cfg {cmd[3]} --dist-url {cmd[4]}"

        p = subprocess.Popen(res, shell=True)
        return_code = p.wait()
        write_log(cmd, return_code)


if __name__ == '__main__':
    main()