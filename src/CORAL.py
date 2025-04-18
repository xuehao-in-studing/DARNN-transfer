import torch
import torch.nn as nn

class CORAL(nn.Module):
    """
    Implementation of CORAL (Correlation Alignment) loss for PyTorch.
    Aligns second-order statistics (covariance) of source and target features.
    Reference: Sun, Baochen, and Kate Saenko. "Deep CORAL: Correlation Alignment for Deep Domain Adaptation." ECCV 2016.
    """
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize (bool): If True, divide loss by 4 * d^2 as in original paper.
                              If False, use raw Frobenius norm squared of covariance difference.
        """
        super(CORAL, self).__init__()
        self.normalize = normalize

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source (Tensor): activations from source domain, shape (batch_size, feature_dim)
            target (Tensor): activations from target domain, shape (batch_size, feature_dim)
        Returns:
            coral_loss (Tensor): scalar tensor
        """
        # Check dimensions
        b_s, d = source.size()
        b_t, _ = target.size()

        # Compute covariance of source
        # subtract mean per feature
        source_mean = source.mean(dim=0, keepdim=True)
        source_centered = source - source_mean
        # covariance: C_s = (X_s^T X_s) / (n_s - 1)
        cov_s = (source_centered.t() @ source_centered) / (b_s - 1)

        # Compute covariance of target
        target_mean = target.mean(dim=0, keepdim=True)
        target_centered = target - target_mean
        cov_t = (target_centered.t() @ target_centered) / (b_t - 1)

        # Compute Frobenius norm between covariances
        diff = cov_s - cov_t
        loss = torch.sum(diff * diff)

        if self.normalize:
            loss = loss / (4 * d * d)

        return loss

# Example usage:
if __name__ == "__main__":
    # random example
    src = torch.randn(32, 64)  # batch 32, feature dim 64
    tgt = torch.randn(32, 64)
    coral = CORAL()
    loss_value = coral(src, tgt)
    print(f"CORAL loss: {loss_value.item():.6f}")
