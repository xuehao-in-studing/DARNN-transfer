import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.arguments import parse_args
from src.load_data import load_data
from src.mmd import linear_mmd2
from src.model import DANN_with_DALSTM
from src.utils import setup_seed

# Visualization of the domain classifier and regressor

# 读取model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"


def get_model(args):
    """Get model and data."""
    # Initialize model
    setup_seed(args.seed)
    print("==> Load dataset ...")
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
                             args.batchsize, args.lr, args.epochs)
    model = model.to(device)
    model_path = f'../models/{args.targetdomain}_{args.object_col}.pt'
    model.load_state_dict(torch.load(model_path))
    list_src, test_data_trg = list(enumerate(data_src)), list(enumerate(test_data_trg))
    return model, list_src, test_data_trg


def hook_fn(model, input, output):
    # 存储特征
    global features
    features = output.detach().cpu()


if __name__ == '__main__':
    args = parse_args()
    model, list_src, test_data_trg = get_model(args)
    # 注册hook
    hook = model.feature_extractor.register_forward_hook(hook_fn)
    src_features = torch.Tensor()
    tar_features = torch.Tensor()
    for i, (x_src, y_src_prev, y_src_true) in list_src:
        x_src, y_src_prev, y_src_true = x_src.to(device), y_src_prev.to(device), y_src_true.to(device)
        model(x_src, y_src_prev, 0.5)
        # cat
        src_features = torch.cat((src_features, features), 0)
    for i, (x_tar, y_tar_prev, y_tar_true) in test_data_trg:
        x_tar, y_tar_prev, y_tar_true = x_tar.to(device), y_tar_prev.to(device), y_tar_true.to(device)
        model(x_tar, y_tar_prev, 0.5)
        # cat
        tar_features = torch.cat((tar_features, features), 0)
    hook.remove()
    # 提取源域和目标域的特征
    src_features = src_features.squeeze().numpy()
    tar_features = tar_features.squeeze().numpy()
    # # 使用T-SNE降维到2D
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2)
    source_features_2d = tsne.fit_transform(src_features)
    target_features_2d = tsne.fit_transform(tar_features)

    len_src = int(source_features_2d.shape[0] / target_features_2d.shape[0])
    # src 截断为与 tar 一样的长度
    mmd = 0
    # 计算KL散度
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    kl = 0

    for i in range(len_src - 1):
        source_feature = src_features[i * tar_features.shape[0]:(i + 1) * tar_features.shape[0], :]
        mmd += linear_mmd2(torch.FloatTensor(source_feature), torch.FloatTensor(tar_features))
        kl += kl_loss(torch.log_softmax(torch.FloatTensor(source_feature), dim=1),
                      torch.softmax(torch.FloatTensor(tar_features), dim=1))
    print(f"mmd: {mmd}")
    print(f"kl: {kl}")

    # 绘制源域和目标域的特征分布图
    plt.figure(figsize=(8, 6), dpi=480)
    plt.scatter(source_features_2d[:, 0], source_features_2d[:, 1], label='Source Domain', alpha=0.8, s=10)
    plt.scatter(target_features_2d[:, 0], target_features_2d[:, 1], label='Target Domain', alpha=0.8, s=10)
    plt.title('Feature Distribution Before Domain Adaptation')
    plt.legend()
    plt.savefig('../plots/feature_distribution_before_domain_adaptation.png')
    plt.show()
