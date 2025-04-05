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


class DANN_with_DALSTM(nn.Module):
    def __init__(self, X, y, T, num_hidden, batch_size, learning_rate=1e-3, epochs: int = 200,
                 model_path: str = None,
                 parallel: bool = False,
                 ):
        super(DANN_with_DALSTM, self).__init__()

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

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.encoder_num_hidden + self.decoder_num_hidden, 2),
            # nn.ReLU(),
            # nn.LogSoftmax(dim=1)
        )

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
        self.early_stopping = EarlyStopping(patience=5, verbose=True,path=model_path)

    def forward(self, X, y_prev, alpha):
        # Extract features from source and target using DA-RNN Encoder
        feature = self.feature_extractor(X, y_prev)

        reverse_feature = ReverseLayerF.apply(feature, alpha)

        # Domain Adaptation: Forward pass through the domain classifier (for adversarial loss)
        domain_pred = self.domain_classifier(reverse_feature.view(feature.size(0), -1))

        # Classification: Forward pass through the final task classifier
        val_pred = self.regressor(feature.view(feature.size(0), -1))

        return val_pred, domain_pred


if __name__ == '__main__':
    batch = 32
    T = 10
    input_size = 20
    num_hidden = 32
    dummy_X_src = torch.rand(batch, T, input_size)
    dummy_X_tar = torch.rand(batch, T, input_size)
    dummy_y_prev_src = torch.rand(batch, T - 1)
    dummy_y_prev_tar = torch.rand(batch, T - 1)
    model = DANN_with_DALSTM(dummy_X_src, dummy_y_prev_src, T, num_hidden, batch)
    # torchinfo
    from torchinfo import summary

    summary(model, input_size=[(batch, T, input_size), (batch, T, input_size), (batch, T - 1,), (batch, T - 1,)])
    # model(dummy_X_src, dummy_X_tar, dummy_y_prev_src, dummy_y_prev_tar)
