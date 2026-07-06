"""
@time: 2026/02/04
@file: split_dataset.py
@author: WD                     ___       __   ________            
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ / 
                                ____/|__/      /_____/  
"""
import cv2 as cv
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import os
import scipy.io as scio
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import re
import yaml
from tqdm import tqdm
import random

def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data

def show_config(params):
    print('Split Result:')
    print('-' * 70)
    print('|%40s | %25s|' % ('Categories', 'Numbers'))
    print('-' * 70)
    total_train_num, total_test_num = 0, 0
    for key, value in params.items():
        print('|%40s | %25s|' % (str(key), str(value)))
        if '[train]' in key:
            total_train_num += value
        elif '[test]' in key:
            total_test_num += value
    print('-' * 70)
    print('|%40s | %25s|' % ('Total Train', str(total_train_num)))
    print('|%40s | %25s|' % ('Total Test', str(total_test_num)))
    print('-' * 70)

def statistical_number_class(data_root):
    cls_num_dir = {}
    cls_list = os.listdir(data_root)
    print(cls_list)
    for cls in cls_list:
        img_files = os.listdir(os.path.join(data_root, cls, 'BMPImages'))
        grass_idxs = [i for i, s in enumerate(img_files) if 'Grass_KU' in s]
        road_idxs = [i for i, s in enumerate(img_files) if 'Road_KU' in s]
        cls_num_dir[cls + ' [Grass]'] = len(grass_idxs)
        cls_num_dir[cls + ' [Road]'] = len(road_idxs)
    print('---------------------')
    print('统计数量如下: ')
    show_config(cls_num_dir)
    # for k, v in cls_num_dir.items():
    #     print('{}------------\t {}'.format(k, v))
        

def parse_filename_info(filename):
    """
    Parse SAR image attribute information from the filename
    Args:
        filename (str): The filename of the SAR image.
    Returns:
        dict: A dictionary containing the parsed image attributes.
    """
    info = {
        'scene': None,
        'angle': None,
        'pitch': None,
        'band': None,
        'polar': None
    }
    img_name = Path(filename).stem
    # Grass_KU_10_001_0.0_look_1_234_238_61.515_Pulse_24260_25699_HV_Bulldozer(Lift)_256
    img_name_ = img_name.split('_')
    info['scene'] = img_name_[0]
    info['band'] = img_name_[1]
    info['pitch'] = float(img_name_[2])
    info['angle'] = float(img_name_[4])
    info['polar'] = img_name_[13]
    info['cls'] = img_name.split('_' + info['polar'] + '_')[-1][:-4]
    info['name'] = img_name
    return info

def check_match_conditions(info, conditions):
    """
    Check if the image information matches the configuration conditions
    Args:
        info (dict): A dictionary containing the parsed image attributes.
        conditions (dict): A dictionary containing the configuration conditions.
    Returns:
        bool: True if the image matches the conditions, False otherwise.
    """    
    if info['angle'] is not None:
        if not (conditions['angle'][0] <= info['angle'] <= conditions['angle'][1]):
            return False
    else:
        print(f"Warning: angle not found in image name {info['name']}, skip check")
        return False
    
    if info['pitch'] is not None:
        if info['pitch'] not in conditions['pitch']:
            return False
    else:
        print(f"Warning: pitch not found in image name {info['name']}, skip check")
        return False
    
    if info['band'] is not None and conditions.get('band'):
        if info['band'] not in conditions['band']:
            return False
    else:
        print(f"Warning: band not found in image name {info['name']}, skip check")
        return False
    
    if info['scene'] is not None and conditions.get('scene'):
        if info['scene'] not in conditions['scene']:
            return False
    else:
        print(f"Warning: scene not found in image name {info['name']}, skip check")
        return False
    
    if info['polar'] is not None and conditions.get('polar'):
        if info['polar'] not in conditions['polar']:
            return False
    else:
        print(f"Warning: polar not found in image name {info['name']}, skip check")
        return False
    
    return True


def split_dataset_txt(yaml_path, test_ratio=0.3, train_few_ratio=1.0):
    """
    Split SAR dataset according to YAML configuration.
    Images are divided into three groups:
    1. Satisfy only train settings -> add to training set
    2. Satisfy only test settings -> add to test set
    3. Satisfy both settings -> split by test_ratio
    Args:
        yaml_path (str): The path to the YAML configuration file.
        test_ratio (float, optional): The ratio of test samples. Defaults to 0.3.
    """
    cfg = yaml_load(yaml_path)
    data_root = cfg['data_path']
    train_txt_path = cfg['train_txt_path']
    test_txt_path = cfg['test_txt_path']
    train_config = cfg['sar_train']
    test_config = cfg['sar_test']
    class_names = cfg['names']
    
    random.seed(0)
    
    if os.path.exists(train_txt_path):
        os.remove(train_txt_path)
    if os.path.exists(test_txt_path):
        os.remove(test_txt_path)
    
    cls_num_dir = {}
    cls_list = list(class_names.values())
    
    for cls in tqdm(cls_list, desc='Processing classes'):
        cls_dir = os.path.join(data_root, cls, 'BMPImages')
        if not os.path.exists(cls_dir):
            print(f'Warning: {cls_dir} not found, skipping...')
            continue
        
        img_paths = list(pathlib.Path(cls_dir).glob('*.bmp'))
        
        only_train_paths = []
        only_test_paths = []
        both_paths = []
        
        for img_path in img_paths:
            info = parse_filename_info(img_path)
            match_train = check_match_conditions(info, train_config)
            match_test = check_match_conditions(info, test_config)
            
            if match_train and match_test:
                both_paths.append(img_path)
            elif match_train:
                only_train_paths.append(img_path)
            elif match_test:
                only_test_paths.append(img_path)
        
        random.shuffle(both_paths)
        split_idx = round(len(both_paths) * (1 - test_ratio))
        train_from_both = both_paths[:split_idx]
        test_from_both = both_paths[split_idx:]
        
        # print('train_from_both: ', len(train_from_both))
        # print('test_from_both: ', len(test_from_both))
        print('Only_train_paths: ', len(only_train_paths))
        train_paths = only_train_paths + train_from_both
        test_paths = only_test_paths + test_from_both
        
        random.shuffle(train_paths)
        random.shuffle(test_paths)
        
        if train_few_ratio != 1.0:
            train_paths = train_paths[:round(len(train_paths)*train_few_ratio)]
        
        # cls_num_dir[cls + ' [only_train]'] = len(only_train_paths)
        # cls_num_dir[cls + ' [only_test]'] = len(only_test_paths)
        # cls_num_dir[cls + ' [both->train]'] = len(train_from_both)
        # cls_num_dir[cls + ' [both->test]'] = len(test_from_both)
        cls_num_dir[cls + ' [train]'] = len(train_paths)
        cls_num_dir[cls + ' [test]'] = len(test_paths)

        with open(train_txt_path, 'a') as f:
            for train_path in train_paths:
                f.write('/'.join(train_path.parts[-3:]) + '\n')
        
        with open(test_txt_path, 'a') as f:
            for test_path in test_paths:
                f.write('/'.join(test_path.parts[-3:]) + '\n')
    
    show_config(cls_num_dir)
    print(f'\nTrain txt saved to: {train_txt_path}')
    print(f'Test txt saved to: {test_txt_path}')




def split_azimuth_dataset_txt(yaml_path, test_ratio=0.3, train_azimuth_interval=10):
    """
    Split SAR dataset according to YAML configuration.
    Images are divided into three groups:
    1. Satisfy only train settings -> add to training set
    2. Satisfy only test settings -> add to test set
    3. Satisfy both settings -> split by test_ratio
    Args:
        yaml_path (str): The path to the YAML configuration file.
        test_ratio (float, optional): The ratio of test samples. Defaults to 0.3.
    """
    cfg = yaml_load(yaml_path)
    data_root = cfg['data_path']
    train_txt_path = cfg['train_txt_path']
    test_txt_path = cfg['test_txt_path']
    train_config = cfg['sar_train']
    test_config = cfg['sar_test']
    class_names = cfg['names']
    
    random.seed(0)
    
    if os.path.exists(train_txt_path):
        os.remove(train_txt_path)
    if os.path.exists(test_txt_path):
        os.remove(test_txt_path)
    
    cls_num_dir = {}
    cls_list = list(class_names.values())
    
    for cls in tqdm(cls_list, desc='Processing classes'):
        cls_dir = os.path.join(data_root, cls, 'BMPImages')
        if not os.path.exists(cls_dir):
            print(f'Warning: {cls_dir} not found, skipping...')
            continue
        
        img_paths = list(pathlib.Path(cls_dir).glob('*.bmp'))
        
        only_train_paths = []
        only_test_paths = []
        both_paths = []
        train_HH_paths = []
        train_HV_paths = []
        train_VH_paths = []
        train_VV_paths = []
        

        for img_path in img_paths:
            info = parse_filename_info(img_path)
            match_train = check_match_conditions(info, train_config)
            match_test = check_match_conditions(info, test_config)
            
            if match_train and match_test:
                both_paths.append(img_path)
            elif match_train:
                if info['polar'] == 'HH':
                    train_HH_paths.append(img_path)
                if info['polar'] == 'HV':
                    train_HV_paths.append(img_path)
                if info['polar'] == 'VH':
                    train_VH_paths.append(img_path)
                if info['polar'] == 'VV':
                    train_VV_paths.append(img_path)
                only_train_paths.append(img_path)
            elif match_test:
                only_test_paths.append(img_path)
        
        random.shuffle(both_paths)
        split_idx = round(len(both_paths) * (1 - test_ratio))
        train_from_both = both_paths[:split_idx]
        test_from_both = both_paths[split_idx:]
        
        # print('train_from_both: ', len(train_from_both))
        # print('test_from_both: ', len(test_from_both))
        print('Only_train_paths: ', len(only_train_paths))
        train_paths = only_train_paths + train_from_both
        test_paths = only_test_paths + test_from_both
        
        
        # random.shuffle(train_paths)
        # random.shuffle(test_paths)
        print(len(train_paths))
        interval_train_path = []
        if train_azimuth_interval != 1:
            for train_polar in [train_HH_paths, train_HV_paths, train_VH_paths, train_VV_paths]:
                interval_train_path = interval_train_path + train_polar[::train_azimuth_interval]

            print("-----------------------", len(interval_train_path))
            cls_num_dir[cls + ' [train]'] = len(interval_train_path)
            cls_num_dir[cls + ' [test]'] = len(test_paths)
        else:
        # cls_num_dir[cls + ' [only_train]'] = len(only_train_paths)
        # cls_num_dir[cls + ' [only_test]'] = len(only_test_paths)
        # cls_num_dir[cls + ' [both->train]'] = len(train_from_both)
        # cls_num_dir[cls + ' [both->test]'] = len(test_from_both)
            cls_num_dir[cls + ' [train]'] = len(train_paths)
            cls_num_dir[cls + ' [test]'] = len(test_paths)

        with open(train_txt_path, 'a') as f:
            if train_azimuth_interval!=1:
                for train_path in interval_train_path:
                    f.write('/'.join(train_path.parts[-3:]) + '\n')
            else:
                for train_path in train_paths:
                    f.write('/'.join(train_path.parts[-3:]) + '\n')
        
        with open(test_txt_path, 'a') as f:
            for test_path in test_paths:
                f.write('/'.join(test_path.parts[-3:]) + '\n')
    
    show_config(cls_num_dir)
    print(f'\nTrain txt saved to: {train_txt_path}')
    print(f'Test txt saved to: {test_txt_path}')
if __name__ == '__main__':
    # split_dataset_soc_txt('J:/CSAR-ATR/SOC', 'E:/SAR_Code/CSAR_ATR_Bench')

    # statistical_number_class('J:/CSAR-ATR/SOC')
    cfg = yaml_load('./Split_Config/EOC_fewtrain_depression_40.yaml')
    print(cfg['sar_train'])
    print(cfg['sar_test'])
    # split_dataset_txt('./Pretrain/cfg/datasets/Pretrain_SOC_Semi_trailer_Towing_Truck.yaml', test_ratio=0.0)

    split_azimuth_dataset_txt('./Split_Config/EOC_fewtrain_depression_40.yaml', test_ratio=0.0, train_azimuth_interval=1)