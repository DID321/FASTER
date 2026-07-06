"""
@time: 2026/03/10
@file: confusion_matrix.py
@author: WD                     ___       __   ________
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ /
                                ____/|__/      /_____/

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Classification.utils import LOGGER
import os

rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
def cal_confusion_matrix(preds, gts, num_classes, is_norm=True):
    x = np.stack([gts, preds], axis=1)
    # define confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i, j in x:
        confusion_matrix[i, j] = confusion_matrix[i, j] + 1
    if is_norm:
       confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    return confusion_matrix


def plot_confusion_matrix(cm, class_names, save_path):
    fig = plt.figure()
    prob_matrix = np.around((cm/np.sum(cm, 1)), 3)
    tick_marks = np.arange(len(class_names))
    plt.imshow(prob_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(tick_marks, class_names, rotation=45, horizontalalignment='right', family='Times New Roman')
    plt.yticks(tick_marks, class_names, family='Times New Roman')
    for i in range(len(prob_matrix)):
        for j in range(len(prob_matrix)):
            if j == i:
                plt.annotate("{:.2f}".format(round(prob_matrix[i, j],4) * 100), xy=(i, j), horizontalalignment='center', verticalalignment='center', family='Times New Roman', color='white', fontsize=8, fontweight='bold')
            else:
                plt.annotate("{:.2f}".format(round(prob_matrix[i, j],4) * 100), xy=(j, i), horizontalalignment='center', verticalalignment='center', family='Times New Roman', fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label', family='Times New Roman', fontsize=17, fontweight='bold')
    plt.xlabel('Predicted label', family='Times New Roman', fontsize=17, fontweight='bold')
    plt.tight_layout()
    #plt.savefig('xxx.pdf', bbox_inches='tight')
    plt.savefig(save_path, bbox_inches='tight', dpi=400)

def show_save_acc(acc1, acc5, cm_nrom, class_names, save_path):
    acc_dir = {}
    for i, name in enumerate(class_names):
        acc_dir[name] = cm_nrom[i, i]*100
    avg_acc = np.sum(np.array(list(acc_dir.values()))) / len(class_names)
    LOGGER.info('Accuracy Result:')
    LOGGER.info('-' * 70)
    LOGGER.info('|%30s | %25s|' % ('class name', 'accuracy'))
    LOGGER.info('-' * 70)
    for key, value in acc_dir.items():
        LOGGER.info('|%30s | %25s|' % (str(key), str(value)))
    LOGGER.info('-' * 70)
    LOGGER.info('|%30s | %25s|' % ('Mean accuracy', str(avg_acc)))
    LOGGER.info('-' * 70)
    LOGGER.info('|%30s | %25s|' % ('Top1 accuracy', str(acc1)))
    LOGGER.info('-' * 70)
    LOGGER.info('|%30s | %25s|' % ('Top5 accuracy', str(acc5)))
    LOGGER.info('-' * 70)
    if os.path.exists(save_path):
        os.remove(save_path)

    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(f"|{'Accuracy Result':^60}|\n")
        f.write('-' * 70 + '\n')
        f.write(f"|{'class name':^30} | {'accuracy':^25}|\n")
        f.write('-' * 70 + '\n')
        for key, value in acc_dir.items():
            f.write(f"|{str(key):>30} | {str(value):>25}|\n")
        f.write('-' * 70 + '\n')
        f.write(f"|{'Mean Accuracy':>30} | {str(avg_acc):>25}|\n")
        f.write('-' * 70 + '\n')
        f.write(f"|{'Top1 Accuracy':>30} | {str(acc1):>25}|\n")
        f.write('-' * 70 + '\n')
        f.write(f"|{'Top5 Accuracy':>30} | {str(acc5):>25}|\n")
        f.write('-' * 70 + '\n')
