# utils.py
import torch
import torch.nn.functional as F

# Note: You might need to install a library for focal loss, e.g., pip install focal-loss-torch
# Or implement it yourself. Here's a placeholder.
def dice_loss(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return 1 - dice

def focal_loss(preds, targets, alpha=0.25, gamma=2.0):
    # This is a placeholder; use a proper implementation
    bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    f_loss = alpha * (1-pt)**gamma * bce_loss
    return f_loss.mean()

class CombinedLoss(torch.nn.Module):
    def __init__(self, focal_weight=20.0):
        super().__init__()
        self.focal_weight = focal_weight
    
    def forward(self, preds, targets):
        return self.focal_weight * focal_loss(preds, targets) + dice_loss(preds, targets)