
import torch
from torch import nn
from torch.nn import Conv3d, BatchNorm3d, ReLU
from torch.nn.functional import interpolate
from prepare import prepare
from train import train
from metrics import MyDiceLoss

# Define ResNet and Bottleneck classes
class ResNet(nn.Module):
    def __init__(self, in_filters):
        super(ResNet, self).__init__()
        self.seq = nn.Sequential(
            ReLU(),
            BatchNorm3d(in_filters),
            Conv3d(in_filters, in_filters, 3, padding=1),
            ReLU(),
            BatchNorm3d(in_filters),
            Conv3d(in_filters, in_filters, 3, padding=1)
        )
    def forward(self, x):
        return x + self.seq(x)

class Bottleneck(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(Bottleneck, self).__init__()
        self.seq = nn.Sequential(
            ReLU(),
            BatchNorm3d(in_filters),
            Conv3d(in_filters, out_filters, 1)
        )
    def forward(self, x):
        return self.seq(x)

# Define RatLesNetv2 model
class RatLesNetv2(nn.Module):
    def __init__(self, modalities, filters):
        super(RatLesNetv2, self).__init__()
        self.conv1 = Conv3d(modalities, filters, 1)
        self.block1 = ResNet(filters)
        self.mp1 = nn.MaxPool3d(2, ceil_mode=True)
        self.block2 = ResNet(filters)
        self.mp2 = nn.MaxPool3d(2, ceil_mode=True)
        self.block3 = ResNet(filters)
        self.mp3 = nn.MaxPool3d(2, ceil_mode=True)
        self.bottleneck1 = Bottleneck(filters, filters)
        self.block4 = ResNet(filters * 2)
        self.bottleneck2 = Bottleneck(filters * 2, filters)
        self.block5 = ResNet(filters * 2)
        self.bottleneck3 = Bottleneck(filters * 2, filters)
        self.block6 = ResNet(filters * 2)
        self.bottleneck4 = Bottleneck(filters * 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        block1_out = self.block1(x)
        block1_size = block1_out.size()

        x = self.mp1(block1_out)
        block2_out = self.block2(x)
        block2_size = block2_out.size()

        x = self.mp2(block2_out)
        block3_out = self.block3(x)
        block3_size = block3_out.size()

        x = self.mp3(block3_out)
        b1 = self.bottleneck1(x)

        x = interpolate(b1, block3_size[2:], mode="trilinear")
        x = torch.cat([x, block3_out], dim=1)
        block4_out = self.block4(x)

        b2 = self.bottleneck2(block4_out)
        x = interpolate(b2, block2_size[2:], mode="trilinear")
        x = torch.cat([x, block2_out], dim=1)

        block5_out = self.block5(x)
        b3 = self.bottleneck3(block5_out)
        x = interpolate(b3, block1_size[2:], mode="trilinear")
        x = torch.cat([x, block1_out], dim=1)

        block6_out = self.block6(x)
        b4 = self.bottleneck4(block6_out)
        return b4

# Initialize the model
filters = 32
modalities = 1
device = torch.device('cuda')
model = RatLesNetv2(modalities, filters).to(device)

# Load pre-trained weights
model.load_state_dict(torch.load('/content/RatLesNetv2-alldata.model'))

# Freeze the encoder layers
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.block1.parameters():
    param.requires_grad = False
for param in model.mp1.parameters():
    param.requires_grad = False
for param in model.block2.parameters():
    param.requires_grad = False
for param in model.mp2.parameters():
    param.requires_grad = False
for param in model.block3.parameters():
    param.requires_grad = False
for param in model.mp3.parameters():
    param.requires_grad = False
for param in model.bottleneck1.parameters():
    param.requires_grad = False

# Enable gradient computation for decoder layers
for param in model.block4.parameters():
    param.requires_grad = True
for param in model.bottleneck2.parameters():
    param.requires_grad = True
for param in model.block5.parameters():
    param.requires_grad = True
for param in model.bottleneck3.parameters():
    param.requires_grad = True
for param in model.block6.parameters():
    param.requires_grad = True
for param in model.bottleneck4.parameters():
    param.requires_grad = True

# Training parameters
data_in = prepare('/content/data_folder', cache=True)
num_epochs = 289
loss_function = MyDiceLoss
optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

# Train the model
if __name__ == '__main__':
    for epoch in range(1):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        train(model, data_in, loss_function, optimizer, num_epochs, device)
        print(f"Finished epoch {epoch + 1}/{num_epochs}")
