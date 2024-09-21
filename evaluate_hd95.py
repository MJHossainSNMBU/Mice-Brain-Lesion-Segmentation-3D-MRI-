
import torch
import os
import numpy as np
from medpy.metric import dc
from medpy.metric.binary import hd95
from medpy.metric.binary import hd

# Define the custom Dice Loss function
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

# Loss function
loss = MyDiceLoss

# Load data and initialize test metrics
train_loader, test_loader, val_loader = data_in
save_loss_test1 = []
save_metric_test1 = []
hd95_test1_epoch = []
voxel_spacing = (0.0997441, 0.0997441, 0.5)
countt = 1

# Load the best model checkpoint
model.load_state_dict(torch.load('/content/ModelUnetNoAugment/best_metric_model.pth'))
model = model.to(device)
model.eval()

# Evaluate on test set
with torch.no_grad():
    test1_epoch_loss = 0
    epoch_metric_test1 = 0
    test1_step = 0
    hd95_list_test1 = []

    for test1_data in test_loader:
        test1_step += 1

        test1_volume = test1_data["vol"]
        test1_label = test1_data["seg"]
        test1_label = test1_label != 0
        test1_volume, test1_label = test1_volume.to(device), test1_label.to(device)

        test1_outputs = model(test1_volume)
        test1_loss = loss(test1_outputs, test1_label)
        test1_epoch_loss += test1_loss.item()

        lesion_output = torch.softmax(test1_outputs[0], axis=0).detach().cpu().numpy()
        lesion_output = np.argmax(lesion_output, axis=0)
        lesion_label = test1_label[0].detach().cpu().numpy()
        lesion_label = np.argmax(lesion_label, axis=0)
        test1_metric = dc(lesion_output, lesion_label)
        epoch_metric_test1 += test1_metric

        # Calculate HD95 only if lesions are present
        if np.count_nonzero(lesion_output) != 0 and np.count_nonzero(lesion_label) != 0:
            hd95_test1 = hd95(lesion_output, lesion_label, voxelspacing=voxel_spacing, connectivity=1)
            print(f'Test HD95: {hd95_test1}')
            hd95_list_test1.append(hd95_test1)
        else:
            print("Skipping HD95 calculation for this batch due to no predicted lesions.")
        print(countt)
        countt += 1

# Compute average HD95 for the test epoch
test1_hd95_avg = np.mean(hd95_list_test1)
hd95_test1_epoch.append(test1_hd95_avg)
print(f'Test HD95 average: {test1_hd95_avg}')

# Compute average loss and dice coefficient for the test epoch
test1_epoch_loss /= test1_step
epoch_metric_test1 /= test1_step
print(f'Final test_loss_epoch: {test1_epoch_loss:.4f}')
print(f'Final test_dice_epoch: {epoch_metric_test1:.4f}')
