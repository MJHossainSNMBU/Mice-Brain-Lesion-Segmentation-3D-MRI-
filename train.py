
from monai.utils import first
import matplotlib.pyplot as plt
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm
import torch
import os
from medpy.metric import dc
import nibabel as nib
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from medpy.metric.binary import hd95, hd

# Import MyDiceLoss from metrics.py
from metrics import MyDiceLoss

# Import data preparation functions if needed (assuming data_in is passed from prepare.py)
from prepare import prepare

def train(model, data_in, loss, optim, max_epochs, device):
    '''
    Trains the model for the given number of epochs, and tracks metrics such as Dice score and HD95.
    Includes an early stopping mechanism.
    '''
    test_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    hd95_train_epoch = []
    hd95_test_epoch = []
    voxel_spacing = (0.0997441, 0.0997441, 0.5)
    early_stop_counter = 0
    early_stop_flag = False
    train_loader, test_loader, val_loader = data_in

    for epoch in range(max_epochs):
        model.train()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        count = 1

        for batch_data in train_loader:
            volume = batch_data["vol"]
            label = batch_data["seg"]
            device = torch.device('cuda')
            volume, label = (volume.to(device), label.to(device))

            outputs = model(volume)
            outputs = torch.softmax(outputs, axis=1)
            train_loss = loss(outputs, label)
            
            # Save NIfTI outputs for selected epochs
            if epoch in [1, 30, 60]:
                nib.save(nib.Nifti1Image(volume[0].cpu().detach().numpy(), np.eye(4)), f"im-{epoch}.nii.gz")
                nib.save(nib.Nifti1Image(label[0].cpu().detach().numpy().transpose(1, 2, 3, 0), np.eye(4)), f"label-{epoch}.nii.gz")
                nib.save(nib.Nifti1Image(outputs[0].cpu().detach().numpy().transpose(1, 2, 3, 0), np.eye(4)), f"pred-{epoch}.nii.gz")

            optim.zero_grad()
            train_loss.backward()
            optim.step()

            print(train_loss)
            lesion_output = np.argmax(outputs[0].detach().cpu().numpy(), axis=0)
            lesion_label = np.argmax(label[0].detach().cpu().numpy(), axis=0)
            train_metric = dc(lesion_output, lesion_label)
            print(count)
            print(f'Train_dice: {train_metric:.4f}')
            count += 1

        # Save train metrics and loss
        save_metric_train.append(train_metric)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)
        save_loss_train.append(train_loss.cpu().detach().numpy())
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

        # Learning rate scheduler
        scheduler = StepLR(optim, step_size=1, gamma=0.95)
        scheduler.step()
        print(f"Current learning rate: {optim.param_groups[0]['lr']}")

        if (epoch + 1) % test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_epoch_loss, epoch_metric_test, test_step = 0, 0, 0

                for test_data in val_loader:
                    test_step += 1
                    test_volume = test_data["vol"]
                    test_label = test_data["seg"] != 0
                    test_volume, test_label = test_volume.to(device), test_label.to(device)

                    test_outputs = model(test_volume)
                    test_loss = loss(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()

                    lesion_output = np.argmax(torch.softmax(test_outputs[0], axis=0).detach().cpu().numpy(), axis=0)
                    lesion_label = np.argmax(test_label[0].detach().cpu().numpy(), axis=0)
                    test_metric = dc(lesion_output, lesion_label)
                    epoch_metric_test += test_metric

                test_epoch_loss /= test_step
                epoch_metric_test /= test_step

                print(f'val_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                print(f'val_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                # Save best model and implement early stopping
                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= 20:
                    print('Validation dice loss did not improve in 20 consecutive epochs. Early stopping activated.')
                    early_stop_flag = True
                    break

            if early_stop_flag:
                break
