import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from src.CORAL import CORAL
from src.arguments import parse_args
from src.load_data import load_data
from src.mmd import mix_rbf_mmd2
from src.model import DANN_with_DALSTM, DANN_with_DALSTM_Multi_Src
from src.utils import setup_seed, accuracy, RMSE, get_weights, get_weights_tensor

"""
多源域向单源域迁移
"""


def train(args):
    """Train the model."""
    # Initialize model
    setup_seed(args.seed)
    print("==> Load dataset ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.ntimestep
    # data
    data_src1, src1_X_scaler, src1_Y_scaler = load_data("../data",
                                                        "ZJ", args.batchsize, args.object_col, T)
    data_src2, src2_X_scaler, src2_Y_scaler = load_data("../data",
                                                        "TSJY", args.batchsize, args.object_col, T)
    data_tar, tar_X_scaler, tar_Y_scaler = load_data("../data",
                                                     f"{args.targetdomain}/train", args.batchsize, args.object_col, T)
    test_data_trg, test_tar_X_scaler, test_tar_Y_scaler = load_data("../data",
                                                                    f"{args.targetdomain}/test", args.batchsize,
                                                                    args.object_col, T)

    X = next(iter(data_src1))[0]
    y_prev = next(iter(data_src1))[1]
    coral = CORAL()

    print("==> Initialize DANN_with_DALSTM_Multi_Src model, 多源域实验开始 ...")
    print(f"==> target domain is {args.targetdomain}, object_col is {args.object_col}, device is {device}")
    model = DANN_with_DALSTM_Multi_Src(X, y_prev, args.ntimestep, args.nums_hidden,
                                       args.batchsize, args.lr, args.epochs,
                                       model_path=f'../models/{args.targetdomain}_{args.object_col}.pt')
    model = model.to(device)
    criterion_dis_src1 = nn.CrossEntropyLoss()
    criterion_dis_src2 = nn.CrossEntropyLoss()
    criterion_dis_tar = nn.CrossEntropyLoss()
    criterion_pred_src1 = nn.MSELoss()
    criterion_pred_src2 = nn.MSELoss()
    criterion_pred_tar = nn.MSELoss()
    LAMBDA = args.LAMBDA
    BETA = args.BETA
    iter_per_epoch = len(test_data_trg)
    list_src1, list_src2, list_tar = list(enumerate(data_src1)), list(enumerate(data_src2)), list(enumerate(data_tar))
    list_test_tar = list(enumerate(test_data_trg))
    # Training
    data_src_len_max = max(len(list_src1), len(list_src2))

    data_src_max = data_src1
    data_src = data_src2
    if len(list_src1) < len(list_src2):
        data_src_max = data_src2
        data_src = data_src1
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        batch_j = 0
        test_loss = 0
        test_acc = 0
        p = float(epoch / args.epochs)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # alpha = 2 / (1 + torch.exp(-10 * (epoch) / args.epochs)) - 1
        with tqdm(total=iter_per_epoch, desc=f"Epoch {epoch + 1}/{model.epochs}", position=0,
                  leave=True) as pbar_epoch:
            for batch_id, (x_src1, y_src1_prev, y_src1_true) in enumerate(data_src_max):
                _, (x_tar, y_tar_prev, y_tar_true) = list_tar[batch_j]
                _, (x_src2, y_src2_prev, y_src2_true) = list(enumerate(data_src))[batch_j]
                # Move data to device
                x_src1, x_src2, x_tar = x_src1.to(device), x_src2.to(device), x_tar.to(device)

                y_src1_prev, y_src2_prev, y_tar_prev = y_src1_prev.to(device), y_src2_prev.to(device), y_tar_prev.to(
                    device)
                y_src1_true, y_src2_true, y_tar_true = y_src1_true.to(device), y_src2_true.to(device), y_tar_true.to(
                    device)

                # Zero the gradients
                model.feature_extractor.encoder_optimizer.zero_grad()
                model.feature_extractor.decoder_optimizer.zero_grad()
                model.regressor_optimizer.zero_grad()
                model.domain_classifier_optimizer.zero_grad()

                feature_src1, pred_src1, domain_pred_src1 = model(x_src1, y_src1_prev, alpha)
                feature_src2, pred_src2, domain_pred_src2 = model(x_src2, y_src2_prev, alpha)
                feature_tar, pred_tar, domain_pred_tar = model(x_tar, y_tar_prev, alpha)
                # l1 = mix_rbf_mmd2(feature_src1, feature_tar, [0.1, 0.5, 1.0])
                # l2 = mix_rbf_mmd2(feature_src2, feature_tar, [0.1, 0.5, 1.0])
                # (w1, w2) = get_weights_tensor([l1, l2])
                # 对比实验CORAL
                l1 = coral(feature_src1, feature_tar)
                l2 = coral(feature_src2, feature_tar)
                (w1, w2) = get_weights_tensor([l1, l2])

                # 用0标记为源域，1标记为目标域
                zero_tensor = torch.zeros(domain_pred_src1.shape[0]).long().to(device)
                # 创建一个形状为 (batch_size, 1) 的全一张量
                one_tensor = torch.ones(domain_pred_src1.shape[0]).long().to(device)
                two_tensor = torch.ones(domain_pred_src1.shape[0]).long().to(device) * 2
                loss_dis_src1 = criterion_dis_src1(domain_pred_src1, one_tensor)
                loss_dis_src2 = criterion_dis_src2(domain_pred_src2, two_tensor)
                loss_dis_tar = criterion_dis_tar(domain_pred_tar, zero_tensor)
                loss_pred_src1 = criterion_pred_src1(pred_src1, y_src1_true)
                loss_pred_src2 = criterion_pred_src2(pred_src2, y_src2_true)
                loss_pred_tar = criterion_pred_tar(pred_tar, y_tar_true)

                loss = -LAMBDA * (w1 * loss_dis_src1 + w2 * loss_dis_src2 + loss_dis_tar) + (
                        (1 - BETA) * (w1 * loss_pred_src1 + w2 * loss_pred_src2) + BETA * loss_pred_tar)

                loss.backward()
                model.feature_extractor.encoder_optimizer.step()
                model.feature_extractor.decoder_optimizer.step()
                model.regressor_optimizer.step()
                model.domain_classifier_optimizer.step()
                train_loss += loss.item()
                train_acc += accuracy(y_src1_true, pred_src1)
                batch_j += 1
                if batch_j >= len(list_tar):
                    batch_j = 0
                # pbar_epoch.set_postfix({"Train loss": f"{train_loss:.4f}"})
                # pbar_epoch.update(1)

            # Testing
            model.eval()
            with torch.no_grad():
                for i, (X, y_prev, y_true) in enumerate(test_data_trg):
                    X, y_prev, y_true = X.to(device), y_prev.to(device), y_true.to(device)
                    _, pred_tar, domain_pred_tar = model(X, y_prev, alpha)
                    loss_pred_tar = criterion_pred_tar(pred_tar, y_true)
                    # 测试不在关心域分类损失和源域预测损失
                    # loss = -LAMBDA * (loss_dis_src + loss_dis_tar) + (
                    #     ALPHA * loss_pred_src + (1 - ALPHA) * loss_pred_tar)
                    loss = loss_pred_tar
                    test_loss += loss.item()
                    test_acc += accuracy(y_true, pred_tar)
                    # Set postfix with both train loss and test loss at the end of epoch
                    pbar_epoch.set_postfix({"Train loss": f"{train_loss:.4f}",
                                            "Test loss": f"{test_loss:.4f}",
                                            "Train acc": f"{train_acc / len(data_src1):.4f}",
                                            "Test acc": f"{test_acc / len(test_data_trg):.4f}"})
                    pbar_epoch.update(1)
                    if i == 0:
                        y_preds_plot = pred_tar.detach().cpu().numpy()
                        y_tar_true_plot = y_true.detach().cpu().numpy()
                    else:
                        y_preds_plot = np.vstack((y_preds_plot, pred_tar.detach().cpu().numpy()))
                        y_tar_true_plot = np.vstack((y_tar_true_plot, y_true.detach().cpu().numpy()))
                y_preds_plot = np.vstack(y_preds_plot)
                y_tar_true_plot = np.vstack(y_tar_true_plot)
                # acc
                acc = accuracy(y_preds_plot, y_tar_true_plot)
                # RMSE
                rmse = RMSE(y_preds_plot, y_tar_true_plot)
                # plot
                y_preds_plot = test_tar_Y_scaler.inverse_transform(y_preds_plot)
                y_tar_true_plot = test_tar_Y_scaler.inverse_transform(y_tar_true_plot)
                plt.figure(figsize=(12, 8))
                plt.plot(y_preds_plot, label='Predicted')
                plt.plot(y_tar_true_plot, label="True")
                plt.legend(loc='upper left')
                plt.xlabel('Ring')
                plt.ylabel(f'{args.object_col}')
                plt.title(f'{args.object_col} Prediction, acc: {acc * 100:.2f}%, RMSE: {rmse:.2f}')
                # plt.savefig(f"../plots/{args.object_col}.png")
                # plt.close(fig)
                plt.show()
                print("==> Predict finished")
                # Print training and testing results
                # print(f'Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')
                if model.early_stopping.early_stop:
                    print("Early stopping")
                    break
                model.early_stopping(test_loss, model)
    return model


def test(model, testLoader, args):
    """Test the model."""
    model.eval()
    for i, (X, y_prev, y_true) in enumerate(testLoader):
        y_pred = model.predict(X, y_prev)
        loss = model.criterion(y_pred, y_true)
        print(f'Loss: {loss.item()}')


if __name__ == '__main__':
    args = parse_args()
    train(args)
