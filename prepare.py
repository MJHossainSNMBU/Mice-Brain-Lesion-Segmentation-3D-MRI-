
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    NormalizeIntensityd,
    AsDiscreted,
    EnsureChannelFirstd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from glob import glob
import os

def prepare(in_dir, pixdim=(0.0997441, 0.0997441, 0.5), a_min=-200, a_max=200, spatial_size=[256, 256, 32], cache=False):
    '''
    Prepares the dataset for training, validation, and testing using MONAI transforms.
    
    Args:
    - in_dir (str): Path to the input directory containing images and masks.
    - pixdim (tuple): Pixel dimensions for spacing.
    - a_min (float): Minimum intensity for scaling.
    - a_max (float): Maximum intensity for scaling.
    - spatial_size (list): Desired output spatial size.
    - cache (bool): Whether to use cache for faster data loading.
    
    Returns:
    - train_loader, test_loader, val_loader: Dataloaders for training, testing, and validation datasets.
    '''
    
    # Defining paths for the datasets
    path_train_volumes = sorted(glob(os.path.join(in_dir, "train_image", "*.nii")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "train_mask", "*.nii")))
    path_test_volumes = sorted(glob(os.path.join(in_dir, "test_image", "*.nii")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "test_mask", "*.nii")))
    path_val_volumes = sorted(glob(os.path.join(in_dir, "val_image", "*.nii")))
    path_val_segmentation = sorted(glob(os.path.join(in_dir, "val_mask", "*.nii")))

    # Debugging: Print paths to confirm correct loading
    print("Training Volumes:", path_train_volumes)
    print("Training Segmentations:", path_train_segmentation)
    print("Test Volumes:", path_test_volumes)
    print("Test Segmentations:", path_test_segmentation)
    print("Validation Volumes:", path_val_volumes)
    print("Validation Segmentations:", path_val_segmentation)

    # Creating file dictionaries for each dataset
    train_files = [{"vol": image, "seg": mask} for image, mask in zip(path_train_volumes, path_train_segmentation)]
    test_files = [{"vol": image, "seg": mask} for image, mask in zip(path_test_volumes, path_test_segmentation)]
    val_files = [{"vol": image, "seg": mask} for image, mask in zip(path_val_volumes, path_val_segmentation)]

    # Common transforms for training, testing, and validation
    common_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        AsDiscreted(keys=["seg"], to_onehot=2),
        NormalizeIntensityd(keys=["vol"]),
        ToTensord(keys=["vol", "seg"]),
    ])

    # Create datasets and dataloaders, using cache if enabled
    def create_dataloader(files, transforms, cache):
        if cache:
            dataset = CacheDataset(data=files, transform=transforms, cache_rate=1.0)
        else:
            dataset = Dataset(data=files, transform=transforms)
        return DataLoader(dataset, batch_size=1)

    # Create dataloaders for each set
    train_loader = create_dataloader(train_files, common_transforms, cache)
    test_loader = create_dataloader(test_files, common_transforms, cache)
    val_loader = create_dataloader(val_files, common_transforms, cache)

    return train_loader, test_loader, val_loader
