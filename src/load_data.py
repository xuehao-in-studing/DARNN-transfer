import os.path
from typing import *
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader


def read_my_data(root_dir, domain, object_col: str = None,
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
    input_path = os.path.join(root_dir, domain, f"环切数据_{object_col}.csv").replace("\\", "/")
    df = pd.read_csv(input_path, nrows=250 if debug else None)
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    X = df.loc[:, [x for x in df.columns.tolist() if x != object_col]].to_numpy()
    X = X_scaler.fit_transform(X)
    y = np.array(df.loc[:, object_col]).reshape(-1, 1)
    y = Y_scaler.fit_transform(y)
    joblib.dump(X_scaler, f'../models/{domain.split("/")[0]}_X_scaler.pkl')
    joblib.dump(Y_scaler, f'../models/{domain.split("/")[0]}_{object_col}_scaler.pkl')
    return X, y, X_scaler, Y_scaler


class DALSTMDataset(Dataset):
    """
    多步预测多步训练集，通过调整prediction_length来确定要预测的步数,适用于LSTM
    """

    def __init__(self, X, y, sequence_length, prediction_length):
        """

        :param dataframe: 数据
        :param sequence_length:
        :param prediction_length:
        :param transform: 哪种转换方式
        :param feature_col: 特征列，列表
        :param predict_col: 预测列，列表
        :param ratio:训练集占的比例
        """
        self.X_TRAIN = X
        self.Y_TRAIN = y

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.X_TRAIN) - self.sequence_length - self.prediction_length + 1

    def __getitem__(self, idx):
        # 从 DataFrame 中提取序列数据,
        # 每timestep行数据会作为一个样本，例如1-sequence_length，2-sequence_length+1...
        x = self.X_TRAIN[idx:idx + self.sequence_length]  # 前面的列作为输入特征 XTrain
        y_prev = self.Y_TRAIN[
                 idx: idx + self.sequence_length - 1]  # 'y' 列作为输出标签
        y_true = self.Y_TRAIN[idx + self.sequence_length - 1]
        # 将数据转换为 PyTorch 张量
        x = torch.tensor(x, dtype=torch.float32)
        y_prev = torch.tensor(y_prev, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32)
        # 如果y为二维，转换为一维
        if len(y_prev.shape) == 2:
            y_prev = y_prev.squeeze(-1)
        # shape of x:[Channel,Sequence len]
        return x, y_prev, y_true


def load_data(root_dir, domain, batch_size, object_col="DJ_5", T=10):
    X, y, X_scaler, Y_scaler = read_my_data(root_dir, domain, object_col, debug=False)
    dataSet = DALSTMDataset(X, y, sequence_length=T, prediction_length=1)
    data_loader = DataLoader(dataSet, batch_size=batch_size, shuffle=False, drop_last=True)
    # X,y_prev,y_true = next(iter(data_loader))
    # print(X.shape, y_prev.shape, y_true.shape)
    return data_loader, X_scaler, Y_scaler


if __name__ == '__main__':
    data_src, src_X_scaler, src_Y_scaler = load_data("../data", "ZJ", 32)
    data_tar, tar_X_scaler, tar_Y_scaler = load_data("../data", "HZW/train", 32)
    test_data_trg, test_tar_X_scaler, test_tar_Y_scaler = load_data("../data", "HZW/test", 32)
    for x, y_prev, y_true in data_src:
        print(x.shape, y_prev.shape, y_true.shape)
        print(len(data_src))
        break
    for x, y_prev, y_true in data_tar:
        print(x.shape, y_prev.shape, y_true.shape)
        print(len(data_tar))
        break
    for x, y_prev, y_true in test_data_trg:
        print(x.shape, y_prev.shape, y_true.shape)
        print(len(test_data_trg))
        break

    # for x, y_prev, y_true in data_src:
    #     print(x.shape, y_prev.shape, y_true.shape)
    #     break
