import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator

from src.DALSTM import DALSTM
from src.early_stopping import EarlyStopping
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DaNN_with_DALSTM(nn.Module):
    def __init__(self, X, y, T, num_hidden, batch_size, learning_rate=1e-3, epochs: int = 200,
                 parallel: bool = False,
                 ):
        super(DaNN_with_DALSTM, self).__init__()

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
        self.feature_extractor = DALSTM(self.X, self.y, self.T, self.encoder_num_hidden, self.decoder_num_hidden,
                                        self.batch_size, self.learning_rate, self.epochs,
                                        self.parallel)

        # Domain classifier (for domain adaptation)
        # self.domain_classifier = nn.Sequential(
        #     nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden,
        #               (self.encoder_num_hidden + self.decoder_num_hidden) // 2),
        #     nn.ReLU(),
        #     nn.Linear((self.encoder_num_hidden + self.decoder_num_hidden) // 2, 2)
        #     # Binary classification for domain (source vs target)
        # )
        self.domain_classifier = nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden, 2)

        # Classifier (for final task classification)
        # self.regressor = nn.Sequential(
        #     nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden,
        #               (self.encoder_num_hidden + self.decoder_num_hidden) // 2),
        #     nn.ReLU(),
        #     nn.Linear((self.encoder_num_hidden + self.decoder_num_hidden) // 2, 1)
        #     # Final regressor predictions, outsize is 1 for regression
        # )
        self.regressor = nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden, 1)

        self.regressor_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                  self.regressor.parameters()),
                                                    lr=self.learning_rate)

        self.domain_classifier_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                                          self.domain_classifier.parameters()),
                                                            lr=self.learning_rate)
        self.early_stopping = EarlyStopping(patience=4, verbose=True)

    def forward(self, X_src, X_tar, y_prev_src, y_prev_tar, alpha):
        # Extract features from source and target using DA-RNN Encoder
        feature_src = self.feature_extractor(X_src, y_prev_src)
        feature_tar = self.feature_extractor(X_tar, y_prev_tar)

        reverse_feature_src = ReverseLayerF.apply(feature_src, alpha)
        # reverse_feature_src = reverse_feature_src.permute(1, 0, 2)
        reverse_feature_tar = ReverseLayerF.apply(feature_tar, alpha)
        # reverse_feature_tar = reverse_feature_tar.permute(1, 0, 2)

        # Domain Adaptation: Forward pass through the domain classifier (for adversarial loss)
        domain_pred_src = self.domain_classifier(reverse_feature_src.view(feature_src.size(0), -1))
        domain_pred_tar = self.domain_classifier(reverse_feature_tar.view(feature_tar.size(0), -1))

        # Classification: Forward pass through the final task classifier
        val_pred_src = self.regressor(feature_src.view(feature_src.size(0), -1))
        val_pred_tar = self.regressor(feature_tar.view(feature_tar.size(0), -1))

        return val_pred_src, val_pred_tar, domain_pred_src, domain_pred_tar


if __name__ == '__main__':
    batch = 32
    T = 10
    input_size = 20
    num_hidden = 32
    dummy_X_src = torch.rand(batch, T, input_size)
    dummy_X_tar = torch.rand(batch, T, input_size)
    dummy_y_prev_src = torch.rand(batch, T - 1)
    dummy_y_prev_tar = torch.rand(batch, T - 1)
    model = DaNN_with_DALSTM(dummy_X_src, dummy_y_prev_src, T, num_hidden, batch)
    # torchinfo
    from torchinfo import summary

    summary(model, input_size=[(batch, T, input_size), (batch, T, input_size), (batch, T - 1,), (batch, T - 1,)])
    # model(dummy_X_src, dummy_X_tar, dummy_y_prev_src, dummy_y_prev_tar)
