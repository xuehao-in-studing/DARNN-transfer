"""
Helper functions.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""
import random
import time
from typing import *

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


def read_data(input_path, debug=True):
    """
    Read nasdaq stocks data.

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    df = pd.read_csv(input_path, nrows=250 if debug else None)
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].to_numpy()
    X = X_scaler.fit_transform(X)
    y = np.array(df.NDX).reshape(-1, 1)
    y = Y_scaler.fit_transform(y)
    return X, y, X_scaler, Y_scaler


def read_my_data(input_path, object_col: str = None,
                 debug=True) -> Union[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    读取湛江数据集，其中前面的列均为特征，最后一列为目标值，均为归一化前的值

    Args:
        object_col:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows=250 if debug else None)
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X = df.loc[:, [x for x in df.columns.tolist() if x != object_col]].to_numpy()
    X = X_scaler.fit_transform(X)
    y = np.array(df.loc[:, object_col]).reshape(-1, 1)
    y = Y_scaler.fit_transform(y)
    joblib.dump(X_scaler, '../models/GZ_X_scaler.pkl')
    joblib.dump(Y_scaler, f'../models/GZ_{object_col}_scaler.pkl')
    return X, y, X_scaler, Y_scaler


def accuracy(y_true, y_pred):
    """

    计算模型输出和target之间的acc,只取我们要预测的step进行acc评估
    计算公式:mean(abs((y_pred-y_true)/(y_true+1e-8)))
    :param y_true:
    :param y_pred:
    :return:
    """
    # 防止广播错误
    assert y_true.shape == y_pred.shape
    # 判断类型，如果是 numpy 类型，采用 numpy 的计算方式
    if isinstance(y_true, np.ndarray):
        error = np.abs((y_pred - y_true) / (y_true + 1e-8))
        error_clipped = np.clip(error, 0, 1)  # 限制到 [0, 1]
        return 1 - np.mean(error_clipped)

    # 如果是 PyTorch 张量，采用 PyTorch 的计算方式
    elif isinstance(y_true, torch.Tensor):
        error = torch.abs((y_pred - y_true) / (y_true + 1e-8))
        error_clipped = torch.clamp(error, 0, 1)  # 限制到 [0, 1]
        return 1 - torch.mean(error_clipped)

    else:
        raise TypeError("y should be numpy.ndarray or torch.Tensor")


# 几个评价指标
# RMSE
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


# R2
def R2(y_true, y_pred):
    return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def get_weights(losses):
    """
    数值稳定版本（避免指数下溢）

    :param losses: list[float] 或 np.ndarray
    :return: 权重列表
    """
    losses = np.asarray(losses)
    # 找到最大负损失（等同于最小损失）
    max_neg_loss = np.max(-losses)
    # 计算调整后的指数
    exp_losses = np.exp(-losses - max_neg_loss)
    sum_exp = exp_losses.sum()
    return list(exp_losses / sum_exp)


## 写一个tensor版本的get_weights
def get_weights_tensor(losses):
    """
    数值稳定版本（避免指数下溢）

    :param losses: list[float] 或 np.ndarray
    :return: 权重列表
    """
    losses = torch.tensor(losses)
    # 找到最大负损失（等同于最小损失）
    max_neg_loss = torch.max(-losses)
    # 计算调整后的指数
    exp_losses = torch.exp(-losses - max_neg_loss)
    sum_exp = exp_losses.sum()
    return exp_losses / sum_exp


if __name__ == '__main__':
    # start_time = time.time()
    # time.sleep(5)
    # print(f"Time consumed: {time.time() - start_time} seconds")
    arr = [2, 5, 1, 12, 1]
    arr = np.array(arr)
    weights = get_weights(arr)
    (w1, w2) = get_weights_tensor([1, 2])
    print(w1)
    print(weights)
