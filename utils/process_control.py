import logging
import os
import torch
import argparse
import json
import numpy as np
import sys


def app_path():
    """Returns the base application path."""
    if hasattr(sys, 'frozen'):
        # Handles PyInstaller
        return os.path.dirname(sys.executable).replace("\\", "/")
    return os.path.dirname(__file__).replace("\\", "/")


def get_args(filename='commandline_args.txt'):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(filename, 'r') as f:
        args.__dict__ = json.load(f)
    return args


def get_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    return logging.getLogger(__name__)


def set_environ():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def label_from_output(output):
    _, top_i = output.data.topk(1)
    return top_i[0]


# returns a python float
def to_scalar(var):
    return var.view(-1).data.tolist()[0]


# return the argmax as a python int
def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# 调整优化器的学习率
def adjust_learning_rate(optimizer, t=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= t


# 返回优化器的学习率
def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr


# label转独热编码
def one_hot(y, label_num=2):
    label = torch.LongTensor(np.zeros(label_num)).to(y.device)
    for i in range(label_num):
        if i == float(y[0]):
            label[i] = 1
    return label


def build():
    import PyInstaller.__main__
    # SETUP_PATH = app_path()
    PyInstaller.__main__.run([
        '--name=%s' % "main",  # 生成的exe文件名
        ['--onedir', '--onefile'][0],  # 单个目录 or 单个文件
        '--noconfirm',  # Replace output directory without asking for confimation
        ['--windowed', '--console'][1],
        # '--add-binary=./python3.dll;.',  # 外部的包引入
        # '--add-binary=%s' % SETUP_PATH + '/config/logging.yaml;config',  # 配置项
        # '--add-data=%s' % SETUP_PATH + '/config/config.ini;config',  # 分号隔开，前面是添加路径，后面是添加到哪个目录
        # '--hidden-import=%s' % 'sqlalchemy.ext.baked',
        # '--hidden-import=%s' % 'frozen_dir',  # 手动添加包，用于处理 module not found
        'main.py',  # 入口文件
    ])


def print_prf(result):
    P, R, F1 = [], [], []
    for i in range(len(result[0])):
        t = 0
        for j in range(len(result[1])):
            t += result[i][j]
        if t != 0:
            P.append(result[i][i] / t)
        else:
            P.append('-')
    for j in range(len(result[1])):
        t = 0
        for i in range(len(result[0])):
            t += result[i][j]
        if t != 0:
            R.append(result[j][j] / t)
        else:
            R.append('-')
    print('\n')
    print('P')
    for i in range(len(result[0])):
        if P[i] != '-':
            print(round(P[i], 4), end='\t')
        else:
            print(P[i], end='\t')
    print('\n')
    print('R')
    for j in range(len(result[1])):
        if R[j] != '-':
            print(round(R[j], 4), end='\t')
        else:
            print(R[j], end='\t')
    print('\n')
    print('F1')
    for i in range(len(result[0])):
        if P[i] != '-' and R[i] != '-':
            a = 2 * P[i] * R[i]
            b = P[i] + R[i]
            if b != 0:
                print(round(a / b, 4), end='\t')
            else:
                print('-', end='\t')
        else:
            print('-', end='\t')


def print_f1(result):
    f1 = 0
    P, R, F1 = [], [], []
    for i in range(len(result[0])):
        t = 0
        for j in range(len(result[1])):
            t += result[i][j]
        if t != 0:
            P.append(result[i][i] / t)
        else:
            P.append('-')
    for j in range(len(result[1])):
        t = 0
        for i in range(len(result[0])):
            t += result[i][j]
        if t != 0:
            R.append(result[j][j] / t)
        else:
            R.append('-')
    for i in range(len(result[0])):
        if P[i] != '-' and R[i] != '-':
            a = 2 * P[i] * R[i]
            b = P[i] + R[i]
            if b != 0:
                f1 += a / b
            else:
                pass
        else:
            pass
    return round(f1 / len(result[0]), 4)


"""
def print_f1(result:[[float]]) -> float:
    f1 = 0
    P, R, F1 = [], [], []
    t = 0
    n = len(result[0])
    for i in range(n):
        for j in range(n):
            t += result[i][j]
        if t != 0:
            P.append(result[i][i] / t)
        else:
            P.append('-')
    t = 0
    for j in range(n):
        for i in range(n):
            t += result[i][j]
        if t != 0:
            R.append(result[j][j] / t)
        else:
            R.append('-')
    for i in range(n):
        if P[i] != '-' and R[i] != '-':
            a = 2 * P[i] * R[i]
            b = P[i] + R[i]
            if b != 0:
                f1 += a / b
    return round(f1 / n, 4)
"""


def installer():
    build()


if __name__ == '__main__':
    result = np.array([
        [123., 26., 39., 1., 0.],
        [22., 94., 33., 4., 0.],
        [34., 38., 49., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ])
    print_prf(result)
