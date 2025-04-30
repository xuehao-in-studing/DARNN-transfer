import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator

from src.DALSTM import DALSTM
from src.early_stopping import EarlyStopping
from torch.autograd import Function

from src.utils import orthogonality_loss, orthogonality_loss_multi

"""论文终稿的模型"""


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class FDW_DANN(nn.Module):
    def __init__(self, X, y, T, num_hidden, batch_size, learning_rate=1e-3, epochs: int = 200,
                 model_path: str = None,
                 parallel: bool = False,
                 ):
        super(FDW_DANN, self).__init__()

        self.encoder_num_hidden = num_hidden
        self.decoder_num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T  # input time steps
        self.X = X  # shape: (n_samples, n_features)
        self.y = y  # shape: (n_samples, 1)

        # DA-RNN Feature Extractor
        self.shared_feature_extractor = DALSTM(self.X, self.y, self.T, self.encoder_num_hidden, self.decoder_num_hidden,
                                               self.batch_size, self.learning_rate, self.epochs,
                                               self.parallel)
        self.src1_feature_extractor = DALSTM(self.X, self.y, self.T, self.encoder_num_hidden,
                                             self.decoder_num_hidden, self.batch_size, self.learning_rate, self.epochs,
                                             self.parallel)
        self.src2_feature_extractor = DALSTM(self.X, self.y, self.T, self.encoder_num_hidden,
                                             self.decoder_num_hidden, self.batch_size, self.learning_rate, self.epochs,
                                             self.parallel)
        self.tar_feature_extractor = DALSTM(self.X, self.y, self.T, self.encoder_num_hidden,
                                            self.decoder_num_hidden, self.batch_size, self.learning_rate, self.epochs,
                                            self.parallel)

        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden, 3),
            # nn.ReLU(),
            # nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden, 3),
        )

        self.shared_regressor = nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden, 1)
        self.private_regressor = nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden, 1)

        self.shared_regressor_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                         self.shared_regressor.parameters()),
                                                           lr=self.learning_rate)

        self.domain_discriminator_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                             self.domain_discriminator.parameters()),
                                                               lr=self.learning_rate)

        self.domain_classifier_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                          self.domain_classifier.parameters()),
                                                            lr=self.learning_rate)
        self.private_regressor_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                          self.private_regressor.parameters()),
                                                            lr=self.learning_rate)

        self.early_stopping = EarlyStopping(patience=5, verbose=True, path=model_path)

    def forward(self, X, y_prev, alpha):
        # Extract features from source and target using DA-RNN Encoder
        _shared_feature = self.shared_feature_extractor(X, y_prev)

        _src1_private_feature = self.src1_feature_extractor(X, y_prev)
        _src2_private_feature = self.src2_feature_extractor(X, y_prev)

        _tar_private_feature = self.tar_feature_extractor(X, y_prev)

        reverse_shared_feature = ReverseLayerF.apply(_shared_feature, alpha)

        # Domain Adaptation: Forward pass through the domain classifier (for adversarial loss)
        _domain_pred = self.domain_discriminator(reverse_shared_feature.view(_shared_feature.size(0), -1))

        _src1_domain_class = self.domain_classifier(_src1_private_feature.view(_shared_feature.size(0), -1))
        _src2_domain_class = self.domain_classifier(_src2_private_feature.view(_shared_feature.size(0), -1))
        _tar_domain_class = self.domain_classifier(_tar_private_feature.view(_shared_feature.size(0), -1))

        # src_private_pred = self.private_regressor(src1_private_feature.view(src1_private_feature.size(0), -1))
        _tar_private_pred = self.private_regressor(_tar_private_feature.view(_tar_private_feature.size(0), -1))

        # Classification: Forward pass through the final task classifier
        _val_pred = self.shared_regressor(_shared_feature.view(_shared_feature.size(0), -1))

        return (_val_pred, _domain_pred, _src1_domain_class, _src2_domain_class, _tar_domain_class, _tar_private_pred,
                _shared_feature, _src1_private_feature, _src2_private_feature, _tar_private_feature)


if __name__ == '__main__':
    batch = 32
    T = 10
    input_size = 20
    num_hidden = 32
    dummy_X_src = torch.rand(batch, T, input_size)
    dummy_X_tar = torch.rand(batch, T, input_size)
    dummy_y_prev_src = torch.rand(batch, T - 1)
    dummy_y_prev_tar = torch.rand(batch, T - 1)
    model = FDW_DANN(dummy_X_src, dummy_y_prev_src, T, num_hidden, batch)
    # torchinfo
    from torchinfo import summary

    (val_pred, domain_pred, src1_domain_class, src2_domain_class, tar_domain_class,
     tar_private_pred, shared_feature, src1_private_feature, src2_private_feature, tar_private_feature) = (
        model(dummy_X_src, dummy_y_prev_src, 0.5))

    or_loss = orthogonality_loss_multi(shared_feature,
                                       [src1_private_feature, src2_private_feature], tar_private_feature)

    print(f"shape of or_loss:{or_loss.size()}, val is {or_loss}")
    summary(model, input_size=[(batch, T, input_size), (batch, T - 1,), (batch, T - 1,)])
    # model(dummy_X_src, dummy_X_tar, dummy_y_prev_src, dummy_y_prev_tar)
