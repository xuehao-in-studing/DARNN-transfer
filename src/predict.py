import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from src.FSDANN import FS_DANN
from src.arguments import parse_args
from src.load_data import load_data
from src.model import DANN_with_DALSTM
from src.utils import setup_seed, accuracy, RMSE

# 允许中文和负号
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

col_names_ch = {
    "DJ_5": "掘进速度(mm/min)",
    "DX_DS_1": "盾首水平偏差(mm)",
    "DX_DS_2": "盾首垂直偏差(mm)",
    "DX_DW_1": "盾尾水平偏差(mm)",
    "DX_DW_2": "盾尾垂直偏差(mm)",
}

title_size = 20
label_size = 24
global_font_size = 20
tick_size = 18

def predict(args):
    """Train the model."""
    # Initialize model
    setup_seed(args.seed)
    print("==> Load dataset ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.ntimestep
    # data
    # data_src, src_X_scaler, src_Y_scaler = load_data("../data", "ZJ", args.batchsize)
    # data_tar, tar_X_scaler, tar_Y_scaler = load_data("../data", "HZW/train", args.batchsize)
    test_data_trg, test_tar_X_scaler, test_tar_Y_scaler = load_data("../data", f"{args.targetdomain}/test",
                                                                    args.batchsize, args.object_col, T)

    X = next(iter(test_data_trg))[0]
    y_prev = next(iter(test_data_trg))[1]

    print("==> Initialize DALSTM model ...")
    print(f"==> target domain is {args.targetdomain}, object_col is {args.object_col}")
    model = FS_DANN(X, y_prev, args.ntimestep, args.nums_hidden,
                             args.batchsize, args.lr, args.epochs)
    model_path = f'../models/{args.targetdomain}_{args.object_col}.pt'
    model.load_state_dict(torch.load(model_path), )

    model = model.to(device)

    test_data = list(enumerate(test_data_trg))
    y_true = []
    y_preds = []
    # predict
    for batch_j, (x_tar, y_tar_prev, y_tar_true) in test_data:
        model.eval()
        x_tar, y_tar_prev, y_tar_true = x_tar.to(device), y_tar_prev.to(device), y_tar_true.to(device)
        (pred_tar, domain_pred_tar, _, tar_domain_class, _, tar_private_pred,
         shared_feature, _, tar_private_feature) = model(x_tar, y_tar_prev, 0.5)  ## FSDANN
        # pred_tar, domain_pred_tar = model(x_tar, y_tar_prev, 0.5) ## DANN
        if batch_j == 0:
            y_preds_shared = pred_tar.detach().cpu().numpy()
            y_preds_private = tar_private_pred.detach().cpu().numpy()
            y_preds_plot = (y_preds_private + y_preds_shared) / 2.0
            y_tar_true_plot = y_tar_true.detach().cpu().numpy()
        else:
            y_preds_shared = pred_tar.detach().cpu().numpy()
            y_preds_private = tar_private_pred.detach().cpu().numpy()
            y_preds_plot = np.vstack((y_preds_plot, (y_preds_private + y_preds_shared) / 2.0))
            y_tar_true_plot = np.vstack((y_tar_true_plot, y_tar_true.detach().cpu().numpy()))
        # if batch_j == 0:
        #     # y_preds.append(pred_tar.detach().cpu().numpy())
        #     # y_true.append(y_tar_true.detach().cpu().numpy())
        #     y_preds = pred_tar.detach().cpu().numpy()
        #     y_true = y_tar_true.detach().cpu().numpy()
        # else:
        #     y_preds = np.vstack((y_preds, pred_tar.detach().cpu().numpy()))
        #     y_true = np.vstack((y_true, y_tar_true.detach().cpu().numpy()))
    y_preds = np.vstack(y_preds_plot)
    y_true = np.vstack(y_tar_true_plot)
    # acc
    acc = accuracy(y_preds, y_true)
    print(f"Accuracy: {acc}")
    # RMSE
    rmse = RMSE(y_preds, y_true)
    print(f"RMSE: {rmse}")
    # plot
    y_preds = test_tar_Y_scaler.inverse_transform(y_preds)
    y_true = test_tar_Y_scaler.inverse_transform(y_true)
    x_min = 200
    x_max = len(y_preds) + 200 - 1  # range(start, stop) 生成到 stop-1 结束

    fig = plt.figure(figsize=(14, 6), dpi=720)
    plt.xlim(x_min, x_max)
    plt.plot(range(200, len(y_preds) + 200), y_preds, label='预测值')
    plt.plot(range(200, len(y_preds) + 200), y_true, label="真实值")
    plt.xticks(fontsize=tick_size)  # x轴刻度字体
    plt.yticks(fontsize=tick_size)  # y轴刻度字体
    plt.legend(loc='upper left', fontsize=global_font_size)
    plt.xlabel('环号', fontsize=label_size)
    plt.ylabel(f'{col_names_ch[args.object_col]}', fontsize=label_size)
    # plt.title(f'{args.object_col} Prediction, acc: {acc * 100:.2f}%, RMSE: {rmse:.2f}')
    plt.savefig(f"../plots/{args.object_col}.png")
    plt.close(fig)
    print("==> Predict finished")
    return model


if __name__ == '__main__':
    args = parse_args()
    model = predict(args)
