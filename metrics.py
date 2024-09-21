
import torch

def MyDiceLoss(y_pred, y_true):
    '''
    Custom Dice Loss implementation for PyTorch tensors.
    
    Args:
    - y_pred: Predicted segmentation output.
    - y_true: Ground truth segmentation mask.
    
    Returns:
    - Dice loss value.
    '''
    axis = list([i for i in range(2, len(y_true.shape))])
    num = 2 * torch.sum(y_pred * y_true, axis=axis)
    denom = torch.sum(y_pred + y_true, axis=axis)
    d = (1 - torch.mean(num / (denom + 1e-6)))
    return d
