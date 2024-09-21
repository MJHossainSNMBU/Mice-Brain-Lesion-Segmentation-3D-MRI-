
import os
import torch
from monai.networks.nets import UNet, DynUNet
from prepare import prepare
from train import train
from metrics import MyDiceLoss

# Load the data
data_dir = '/content/data_folder'
data_in = prepare(data_dir, cache=True)

# Create model directory if it doesn't exist
if not os.path.exists('/content/ModelUnetNoAugment'):
    os.makedirs('/content/ModelUnetNoAugment')

model_dir = '/content/ModelUnetNoAugment'

# Set device to GPU
device = torch.device('cuda')

# Define the UNet model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

# Set training parameters
num_epochs = 100
loss_function = MyDiceLoss
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

# Main function to start the training
if __name__ == '__main__':
    for epoch in range(1):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        train(model, data_in, loss_function, optimizer, num_epochs, device)
        print(f"Finished epoch {epoch + 1}/{num_epochs}")
