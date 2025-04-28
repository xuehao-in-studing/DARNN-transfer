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


def orthogonality_loss(H_c_source: torch.Tensor, H_p_source: torch.Tensor,
                       H_c_target: torch.Tensor, H_p_target: torch.Tensor,
                       normalize: bool = False) -> torch.Tensor:
    """
    根据公式计算正交损失:
    L_diff = || H_cs^T H_ps ||_F^2 + || H_ct^T H_pt ||_F^2
    其中 H_cs, H_ps 是源域的公共/私有特征,
    H_ct, H_pt 是目标域的公共/私有特征。

    Args:
        H_c_source (torch.Tensor): 源域公共特征张量, 形状例如 (N, D_cs).
        H_p_source (torch.Tensor): 源域私有特征张量, 形状例如 (N, D_ps).
        H_c_target (torch.Tensor): 目标域公共特征张量, 形状例如 (N, D_ct).
                                   (D_ct 可能等于 D_cs). 第一个维度 N 必须匹配.
        H_p_target (torch.Tensor): 目标域私有特征张量, 形状例如 (N, D_pt).
                                   第一个维度 N 必须匹配.
        normalize (bool, optional): 是否根据批次大小 N 对总损失进行归一化. 默认为 False.

    Returns:
        torch.Tensor: 计算得到的标量损失值.

    Raises:
        ValueError: 如果输入张量不是二维或批次大小 (N) 不匹配.
    """
    # --- 输入检查 ---
    if not (H_c_source.ndim == 2 and H_p_source.ndim == 2 and
            H_c_target.ndim == 2 and H_p_target.ndim == 2):
        raise ValueError("所有输入张量 H_c_source, H_p_source, H_c_target, H_p_target 都必须是二维的")

    batch_size = H_c_source.shape[0]
    if not (H_p_source.shape[0] == batch_size and
            H_c_target.shape[0] == batch_size and
            H_p_target.shape[0] == batch_size):
        raise ValueError(
            f"所有输入张量的第一个维度 (batch_size) 必须相同。 "
            f"获取到的大小为: H_c_source={H_c_source.shape[0]}, "
            f"H_p_source={H_p_source.shape[0]}, H_c_target={H_c_target.shape[0]}, "
            f"H_p_target={H_p_target.shape[0]}"
        )

    # --- 计算源域项: || H_cs^T H_ps ||_F^2 ---
    H_cs_T = H_c_source.t()  # 转置: (N, Dcs) -> (Dcs, N)
    product_s = torch.matmul(H_cs_T, H_p_source)  # 矩阵乘法: (Dcs, N) @ (N, Dps) -> (Dcs, Dps)
    loss_s = torch.sum(product_s ** 2)  # 平方 Frobenius 范数

    # --- 计算目标域项: || H_ct^T H_pt ||_F^2 ---
    H_ct_T = H_c_target.t()  # 转置: (N, Dct) -> (Dct, N)
    product_t = torch.matmul(H_ct_T, H_p_target)  # 矩阵乘法: (Dct, N) @ (N, Dpt) -> (Dct, Dpt)
    loss_t = torch.sum(product_t ** 2)  # 平方 Frobenius 范数

    # --- 总损失 ---
    total_loss = loss_s + loss_t

    # --- 可选的归一化 ---
    if normalize and batch_size > 0:
        total_loss = total_loss / batch_size

    return total_loss


def orthogonality_loss_multi(
        H_c_source_list: List[torch.Tensor],
        H_p_source_list: List[torch.Tensor],
        H_c_target: torch.Tensor,
        H_p_target: torch.Tensor,
        normalize: bool = True
) -> torch.Tensor:
    """
    根据公式计算正交损失:
    L_diff = || H_ct^T H_pt ||_F^2 + Sum_i || H_csi^T H_psi ||_F^2
    其中 H_csi, H_psi 是第 i 个源域的公共/私有特征 (来自列表),
    H_ct, H_pt 是目标域的公共/私有特征。

    Args:
        H_c_source_list (List[torch.Tensor]): 包含 N_s 个 *源域* 公共特征张量的列表。
                                             每个 H_csi 形状例如 (N, D_csi)。
        H_p_source_list (List[torch.Tensor]): 包含 N_s 个 *源域* 私有特征张量的列表。
                                             每个 H_psi 形状例如 (N, D_psi)。
                                             长度必须与 H_c_source_list 相同。
                                             所有张量的 N 必须与 H_c_target 的 N 相同。
        H_c_target (torch.Tensor): *目标域* 公共特征张量, 形状例如 (N, D_ct)。
        H_p_target (torch.Tensor): *目标域* 私有特征张量, 形状例如 (N, D_pt)。
                                     第一个维度 N 必须与 H_c_source_list 中的张量相同。
        normalize (bool, optional): 是否根据批次大小 N 对总损失进行归一化. 默认为 True.

    Returns:
        torch.Tensor: 计算得到的标量损失值.

    Raises:
        ValueError: 如果输入不合法 (列表长度不匹配, 非二维, 批次大小不匹配等).
    """
    # --- 输入检查 ---
    if len(H_c_source_list) != len(H_p_source_list):
        raise ValueError(
            f"源域公共特征列表 H_c_source_list (长度 {len(H_c_source_list)}) 和 "
            f"源域私有特征列表 H_p_source_list (长度 {len(H_p_source_list)}) 必须具有相同的长度。"
        )
    if H_c_target.ndim != 2 or H_p_target.ndim != 2:
        raise ValueError("输入张量 H_c_target 和 H_p_target 必须是二维的。")

    # 使用目标域张量获取参考 batch_size 和设备/类型信息
    batch_size = H_c_target.shape[0]
    if H_p_target.shape[0] != batch_size:
        raise ValueError(
            f"批次大小不匹配。H_c_target 的批次大小为 {batch_size}, 但 H_p_target 的批次大小为 {H_p_target.shape[0]}。")

    # 初始化源域损失总和
    loss_s_sum = torch.tensor(0.0, device=H_c_target.device, dtype=H_c_target.dtype) / len(H_c_source_list)

    # --- 计算目标域项: || H_ct^T H_pt ||_F^2 ---
    H_ct_T = H_c_target.t()  # 转置: (N, Dct) -> (Dct, N)
    product_t = torch.matmul(H_ct_T, H_p_target)  # 矩阵乘法: (Dct, N) @ (N, Dpt) -> (Dct, Dpt)
    loss_t = torch.sum(product_t ** 2)  # 平方 Frobenius 范数

    # --- 计算源域项总和: Sum_i || H_csi^T H_psi ||_F^2 ---
    # 使用 zip 同时迭代两个源域列表
    for i, (H_cs, H_ps) in enumerate(zip(H_c_source_list, H_p_source_list)):
        # 检查每个源域特征对
        if H_cs.ndim != 2 or H_ps.ndim != 2:
            raise ValueError(f"源域列表索引 {i} 处的张量对必须都是二维的。")
        if H_cs.shape[0] != batch_size or H_ps.shape[0] != batch_size:
            raise ValueError(
                f"源域列表索引 {i} 处的批次大小不匹配。 "
                f"期望 {batch_size}, 得到 H_cs={H_cs.shape[0]}, H_ps={H_ps.shape[0]}。"
            )

        # 计算第 i 个源域项: || H_csi^T H_psi ||_F^2
        H_cs_T = H_cs.t()  # 转置: (N, Dcsi) -> (Dcsi, N)
        product_s = torch.matmul(H_cs_T, H_ps)  # 矩阵乘法: (Dcsi, N) @ (N, Dpsi) -> (Dcsi, Dpsi)
        loss_s_term = torch.sum(product_s ** 2)  # 平方 Frobenius 范数

        # 累加源域损失
        loss_s_sum = loss_s_sum + loss_s_term

    # --- 总损失 ---
    total_loss = loss_t + loss_s_sum / len(H_c_source_list)

    # --- 可选的归一化 ---
    if normalize and batch_size > 0:
        total_loss = total_loss / batch_size

    return total_loss


if __name__ == '__main__':
    # start_time = time.time()
    # time.sleep(5)
    # print(f"Time consumed: {time.time() - start_time} seconds")
    # --- 示例用法 ---
    # 假设:
    N = 64
    Dcs1 = 32  # Source 1 common features dim

    # 假设这些张量来自于你的模型不同的分支或不同的输入
    H_cs1_tensor = torch.ones((N, Dcs1), requires_grad=True)
    H_ps1_tensor = torch.zeros((N, Dcs1), requires_grad=True)
    H_cs2_tensor = torch.ones((N, Dcs1), requires_grad=True)
    H_ps2_tensor = torch.zeros((N, Dcs1), requires_grad=True)
    H_ct_tensor = torch.ones((N, Dcs1), requires_grad=True)
    H_pt_tensor = torch.zeros((N, Dcs1), requires_grad=True)

    # 将源域特征放入列表
    source_common_features = [H_cs1_tensor, H_cs2_tensor]
    source_private_features = [H_ps1_tensor, H_ps2_tensor]

    # 使用新函数计算损失
    loss = orthogonality_loss_multi(
        source_common_features, source_private_features,
        H_ct_tensor, H_pt_tensor,
        normalize=True
    )
    print(f"Full Separate Orthogonality Loss: {loss.item()}")

    # 可以进行反向传播
    # loss.backward()
