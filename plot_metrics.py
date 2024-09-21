
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the model directory
model_dir = '/content/ModelUnetNoAugment'

# Load training and validation metrics
train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))

# Set up the plotting area
plt.figure("Monai 3D Unet", figsize=(12, 6))
plt.subplots_adjust(wspace=0.4, hspace=0.5)

# Plot train dice loss
plt.subplot(2, 2, 1)
plt.title("Train dice loss")
x = np.arange(len(train_loss)) + 1
plt.xlabel("epoch")
plt.ylabel("average dice loss")
plt.plot(x, train_loss, color='blue')

# Plot train dice coefficient
plt.subplot(2, 2, 2)
plt.title("Train metric dice coefficient")
plt.xlabel("epoch")
plt.ylabel("average dice coefficient")
plt.plot(x, train_metric, color='green')

# Plot validation dice loss
plt.subplot(2, 2, 3)
plt.title("Validation dice loss")
x = np.arange(len(test_loss)) + 1
plt.xlabel("epoch")
plt.ylabel("average dice loss")
plt.plot(x, test_loss, color='red')

# Plot validation dice coefficient
plt.subplot(2, 2, 4)
plt.title("Validation dice coefficient")
plt.xlabel("epoch")
plt.ylabel("average dice coefficient")
plt.plot(x, test_metric, color='purple')

# Display the plots
plt.show()
