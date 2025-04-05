import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from src.arguments import parse_args
from src.load_data import load_data
from src.model import DANN_with_DALSTM
from src.utils import setup_seed, accuracy, RMSE


def train(args):
    """Train the model."""
    # Initialize model
    setup_seed(args.seed)
    print("==> Load dataset ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = args.ntimestep
    # data
    data_src, src_X_scaler, src_Y_scaler = load_data("../data",
                                                     "ZJ", args.batchsize, args.object_col, T)
    data_tar, tar_X_scaler, tar_Y_scaler = load_data("../data",
                                                     f"{args.targetdomain}/train", args.batchsize, args.object_col, T)
    test_data_trg, test_tar_X_scaler, test_tar_Y_scaler = load_data("../data",
                                                                    f"{args.targetdomain}/test", args.batchsize, args.object_col, T)

    X = next(iter(data_src))[0]
    y_prev = next(iter(data_src))[1]

    print("==> Initialize DALSTM model ...")
    model = DANN_with_DALSTM(X, y_prev, args.ntimestep, args.nums_hidden,
                             args.batchsize, args.lr, args.epochs,
                             model_path=f'../models/{args.targetdomain}_{args.object_col}.pt')
    model = model.to(device)
    criterion_dis_src = nn.CrossEntropyLoss()
    criterion_dis_tar = nn.CrossEntropyLoss()
    criterion_pred_src = nn.MSELoss()
    criterion_pred_tar = nn.MSELoss()
    LAMBDA = args.LAMBDA
    ALPHA = args.ALPHA
    iter_per_epoch = len(test_data_trg)
    list_src, list_tar = list(enumerate(data_src)), list(enumerate(data_tar))
    list_test_tar = list(enumerate(test_data_trg))
    # Training
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
            for batch_id, (x_src, y_src_prev, y_src_true) in enumerate(data_src):
                _, (x_tar, y_tar_prev, y_tar_true) = list_test_tar[batch_j]
                # Move data to device
                x_src, x_tar = x_src.to(device), x_tar.to(device)
                y_src_prev, y_tar_prev = y_src_prev.to(device), y_tar_prev.to(device)
                y_src_true, y_tar_true = y_src_true.to(device), y_tar_true.to(device)

                # Zero the gradients
                model.feature_extractor.encoder_optimizer.zero_grad()
                model.feature_extractor.decoder_optimizer.zero_grad()
                model.regressor_optimizer.zero_grad()
                model.domain_classifier_optimizer.zero_grad()

                pred_src, domain_pred_src = model(x_src, y_src_prev, alpha)
                pred_tar, domain_pred_tar = model(x_tar, y_tar_prev, alpha)

                # 用0标记为源域，1标记为目标域
                zero_tensor = torch.zeros(domain_pred_src.shape[0]).long().to(device)
                # 创建一个形状为 (batch_size, 1) 的全一张量
                one_tensor = torch.ones(domain_pred_src.shape[0]).long().to(device)
                loss_dis_src = criterion_dis_src(domain_pred_src, one_tensor)
                loss_dis_tar = criterion_dis_tar(domain_pred_tar, zero_tensor)
                loss_pred_src = criterion_pred_src(pred_src, y_src_true)
                loss_pred_tar = criterion_pred_tar(pred_tar, y_tar_true)
                loss = -LAMBDA * (loss_dis_src + loss_dis_tar) + (
                        ALPHA * loss_pred_src + (1 - ALPHA) * loss_pred_tar)

                loss.backward()
                model.feature_extractor.encoder_optimizer.step()
                model.feature_extractor.decoder_optimizer.step()
                model.regressor_optimizer.step()
                model.domain_classifier_optimizer.step()
                train_loss += loss.item()
                train_acc += accuracy(y_src_true, pred_src)
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
                    pred_tar, domain_pred_tar = model(X, y_prev, alpha)
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
                                            "Train acc": f"{train_acc / len(data_src):.4f}",
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
