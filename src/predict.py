import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from src.arguments import parse_args
from src.load_data import load_data
from src.model import DANN_with_DALSTM
from src.utils import setup_seed, accuracy, RMSE


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
    test_data_trg, test_tar_X_scaler, test_tar_Y_scaler = load_data("../data", "HZW/test",
                                                                    args.batchsize, args.object_col, T)

    X = next(iter(test_data_trg))[0]
    y_prev = next(iter(test_data_trg))[1]

    print("==> Initialize DALSTM model ...")
    model = DANN_with_DALSTM(X, y_prev, args.ntimestep, args.nums_hidden,
                             args.batchsize, args.lr, args.epochs)
    model_path = f'../models/GZ_{args.object_col}.pt'
    model.load_state_dict(torch.load(model_path))

    model = model.to(device)

    test_data = list(enumerate(test_data_trg))
    y_true = []
    y_preds = []
    # predict
    for batch_j, (x_tar, y_tar_prev, y_tar_true) in test_data:
        model.eval()
        pred_src, pred_tar, domain_pred_src, domain_pred_tar = model(
            x_tar, x_tar, y_tar_prev, y_tar_prev, 0.5)
        if batch_j == 0:
            # y_preds.append(pred_tar.detach().cpu().numpy())
            # y_true.append(y_tar_true.detach().cpu().numpy())
            y_preds = pred_tar.detach().cpu().numpy()
            y_true = y_tar_true.detach().cpu().numpy()
        else:
            y_preds = np.vstack((y_preds, pred_tar.detach().cpu().numpy()))
            y_true = np.vstack((y_true, y_tar_true.detach().cpu().numpy()))
    y_preds = np.vstack(y_preds)
    y_true = np.vstack(y_true)
    # acc
    acc = accuracy(y_preds, y_true)
    print(f"Accuracy: {acc}")
    # RMSE
    rmse = RMSE(y_preds, y_true)
    print(f"RMSE: {rmse}")
    # plot
    y_preds = test_tar_Y_scaler.inverse_transform(y_preds)
    y_true = test_tar_Y_scaler.inverse_transform(y_true)

    fig = plt.figure(figsize=(12, 8), dpi=720)
    plt.plot(y_preds, label='Predicted')
    plt.plot(y_true, label="True")
    plt.legend(loc='upper left')
    plt.xlabel('Ring')
    plt.ylabel(f'{args.object_col}')
    plt.title(f'{args.object_col} Prediction, acc: {acc*100:.2f}%, RMSE: {rmse:.2f}')
    plt.savefig(f"../plots/{args.object_col}.png")
    plt.close(fig)
    print("==> Predict finished")
    return model


if __name__ == '__main__':
    args = parse_args()
    model = predict(args)
