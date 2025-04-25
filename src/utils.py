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


def orthogonality_loss(H_c: torch.Tensor, H_ps: torch.Tensor, H_pt: torch.Tensor,
                       normalize: bool = False) -> torch.Tensor:
    """
    计算正交损失 L_diff = || H_c^T H_ps ||_F^2 + || H_c^T H_pt ||_F^2

    Args:
        H_c (torch.Tensor): 公共特征张量, 形状例如 (N, D_c).
        H_ps (torch.Tensor): 源域私有特征张量, 形状例如 (N, D_ps).
        H_pt (torch.Tensor): 目标域私有特征张量, 形状例如 (N, D_pt).
        normalize (bool, optional): 是否根据批次大小 N 对损失进行归一化. 默认为 False.

    Returns:
        torch.Tensor: 计算得到的标量损失值.

    Raises:
        ValueError: 如果输入张量不是二维或批次大小不匹配.
    """
    # --- 输入检查 ---
    if not (H_c.ndim == 2 and H_ps.ndim == 2 and H_pt.ndim == 2):
        raise ValueError("输入张量 H_c, H_ps, H_pt 都必须是二维的 (例如, batch_size x feature_dim)")

    batch_size = H_c.shape[0]
    if H_ps.shape[0] != batch_size or H_pt.shape[0] != batch_size:
        raise ValueError("所有输入张量的第一个维度 (batch_size) 必须相同。")

    # --- 计算第一项: || H_c^T H_ps ||_F^2 ---
    # 转置 H_c: (N, Dc) -> (Dc, N)
    H_c_T = H_c.t()
    # 矩阵乘法: (Dc, N) @ (N, Dps) -> (Dc, Dps)
    product_s = torch.matmul(H_c_T, H_ps)
    # 计算平方 Frobenius 范数 (所有元素的平方和)
    loss_s = torch.sum(product_s ** 2)

    # --- 计算第二项: || H_c^T H_pt ||_F^2 ---
    # 可以复用 H_c_T
    # 矩阵乘法: (Dc, N) @ (N, Dpt) -> (Dc, Dpt)
    product_t = torch.matmul(H_c_T, H_pt)
    # 计算平方 Frobenius 范数
    loss_t = torch.sum(product_t ** 2)

    # --- 总损失 ---
    total_loss = loss_s + loss_t

    # --- 可选的归一化 ---
    # 有时为了让损失值不受 batch size 影响，会进行归一化
    if normalize and batch_size > 0:
        # 除以 batch size 是一个常见的选择
        total_loss = total_loss / batch_size
        # 或者可以除以乘积矩阵的元素总数，但除以 N 更常见
        # num_elements_s = product_s.numel()
        # num_elements_t = product_t.numel()
        # total_loss = loss_s / num_elements_s + loss_t / num_elements_t

    return total_loss


# --- 如何在 PyTorch 训练循环中使用 ---
# 假设你从模型的不同部分获取了 H_c, H_ps, H_pt 张量
# H_c_features = model.common_encoder(input_data)
# H_ps_features = model.source_private_encoder(source_data)
# H_pt_features = model.target_private_encoder(target_data) # 或者来自同一个模型的不同分支

# 计算损失
# orth_loss = orthogonality_loss(H_c_features, H_ps_features, H_pt_features, normalize=True)

# 将这个损失添加到你的总损失中 (可能需要一个权重系数)
# total_training_loss = main_task_loss + lambda_orth * orth_loss

# 执行反向传播
# total_training_loss.backward()
# optimizer.step()

if __name__ == '__main__':
    # start_time = time.time()
    # time.sleep(5)
    # print(f"Time consumed: {time.time() - start_time} seconds")
    ## 测试orthogonality_loss 函数
    Hc = torch.ones((32, 64))
    Hps = torch.zeros((32, 64))
    Hpt = torch.zeros((32, 64))
    loss = orthogonality_loss(Hc, Hps, Hpt)
    print(loss)
