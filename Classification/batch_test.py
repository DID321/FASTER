import logging
import subprocess
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='test.log')


def write_log(cmd, status):
    if not status:
        logging.info(f'{cmd} test success')
    else:
        logging.info(f'{cmd} test fail')

cmds =[
    # ['VGG16', './logs_100/VGG16_EOC_angle', './cfg/datasets/EOC_angle.yaml', './logs_100/VGG16_EOC_angle/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_angle', './cfg/datasets/EOC_angle_60_120.yaml', './logs_100/VGG16_EOC_angle/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_angle', './cfg/datasets/EOC_angle_120_180.yaml', './logs_100/VGG16_EOC_angle/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_angle', './cfg/datasets/EOC_angle_180_240.yaml', './logs_100/VGG16_EOC_angle/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_angle', './cfg/datasets/EOC_angle_240_300.yaml', './logs_100/VGG16_EOC_angle/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_angle', './cfg/datasets/EOC_angle_300_360.yaml', './logs_100/VGG16_EOC_angle/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_polar', './cfg/datasets/EOC_polar_VH.yaml', './logs_100/VGG16_EOC_polar/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_polar', './cfg/datasets/EOC_polar_VV.yaml', './logs_100/VGG16_EOC_polar/best_model.pth'],

    # ['VGG16', './logs_100/VGG16_EOC_pitch', './cfg/datasets/EOC_pitch.yaml', './logs_100/VGG16_EOC_pitch/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_pitch', './cfg/datasets/EOC_pitch_25_30.yaml', './logs_100/VGG16_EOC_pitch/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_pitch', './cfg/datasets/EOC_pitch_35_40.yaml', './logs_100/VGG16_EOC_pitch/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_pitch', './cfg/datasets/EOC_pitch_45_50.yaml', './logs_100/VGG16_EOC_pitch/best_model.pth'],
    # ['VGG16', './logs_100/VGG16_EOC_pitch', './cfg/datasets/EOC_pitch_55_60.yaml', './logs_100/VGG16_EOC_pitch/best_model.pth'],

    ]
def main():
    for cmd in cmds:
        res = f'python test_SOC.py --model-name {cmd[0]} --log-path {cmd[1]} --data-cfg {cmd[2]} --weights {cmd[3]}'

        p = subprocess.Popen(res, shell=True)
        return_code = p.wait()
        write_log(cmd, return_code)


if __name__ == '__main__':
    main()