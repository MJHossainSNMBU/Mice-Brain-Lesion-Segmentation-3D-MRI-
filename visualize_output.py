
from skimage import measure
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, ToTensord, EnsureChannelFirstd, AsDiscreted, NormalizeIntensityd

# Function to plot slices
def plot_slices(img_path, mask_path, selected_slices, model, model_path, device, row, axes):
    img_transforms = Compose([
        LoadImaged(keys=["vol"]),
        ToTensord(keys=["vol"]),
    ])

    img_data_dict = img_transforms({"vol": img_path})
    img_tensor = img_data_dict["vol"].to(device)
    img_array = img_tensor.cpu().detach().numpy()[:, :, :]

    gt_transforms = Compose([
        LoadImaged(keys=["seg"]),
        ToTensord(keys=["seg"]),
    ])

    gt_data_dict = gt_transforms({"seg": mask_path})
    gt_mask_tensor = gt_data_dict["seg"].to(device)
    gt_mask_array = gt_mask_tensor.cpu().detach().numpy()[:, :, :]

    transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        AsDiscreted(keys=["seg"], to_onehot=2),
        NormalizeIntensityd(keys=["vol"]),
        ToTensord(keys=["vol", "seg"]),
    ])

    data_dict = transforms({"vol": img_path, "seg": mask_path})
    img_tensor = data_dict["vol"].to(device)

    # Load the model
    model_dict = model.state_dict()
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    # Get prediction
    with torch.no_grad():
        model.eval()
        img_tensor = img_tensor.unsqueeze(0)
        pred_tensor = model(img_tensor)
    pred_array = np.argmax(pred_tensor.cpu().detach().numpy()[0], axis=0)

    # Binarize the masks
    gt_mask_array = np.where(gt_mask_array > 0, 1, 0)
    pred_array = np.where(pred_array > 0, 1, 0)

    # Crop the images and masks
    start_idx = (img_array.shape[0] - 180) // 2
    end_idx = start_idx + 180

    img_array = img_array[start_idx:end_idx, start_idx:end_idx, :]
    gt_mask_array = gt_mask_array[start_idx:end_idx, start_idx:end_idx, :]
    pred_array = pred_array[start_idx:end_idx, start_idx:end_idx, :]

    # Rotate arrays
    img_array = np.rot90(img_array, k=-1, axes=(0, 1))
    gt_mask_array = np.rot90(gt_mask_array, k=-1, axes=(0, 1))
    pred_array = np.rot90(pred_array, k=-1, axes=(0, 1))

    for col, i in enumerate(selected_slices):
        ax = axes[row, col]
        ax.imshow(img_array[:, :, i], cmap='gray', alpha=1.0)

        # Plot ground truth mask
        contours_gt = measure.find_contours(gt_mask_array[:, :, i], 0.5)
        for n, contour in enumerate(contours_gt):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='yellow')

        # Plot predicted mask
        contours_pred = measure.find_contours(pred_array[:, :, i], 0.5)
        for n, contour in enumerate(contours_pred):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Slice {i}")

    axes[row, 0].set_ylabel(f"Test Image {row+1}", rotation=90, size='large', labelpad=10)

# Main block to load and visualize slices
if __name__ == '__main__':
    model = RatLesNetv2(modalities, filters).to(device)
    model_path = "/content/drive/MyDrive/Thesis/Final/TL_40images_noaugment_encoder/best_metric_model.pth"

    img_paths = [
        "/content/data_folder/test_image/20170920CH_Exp09_M11.nii",
        "/content/data_folder/test_image/20170628CH_Exp6_ZFPtm1a_M14.nii",
        "/content/data_folder/test_image/20170920CH_Exp09_M12.nii",
        "/content/data_folder/test_image/20170921CH_Exp09_M24.nii",
        "/content/data_folder/test_image/20190320CH_Exp4_M18.nii"
    ]

    mask_paths = [
        "/content/data_folder/test_mask/20170920CH_Exp09_M11_mask.nii",
        "/content/data_folder/test_mask/20170628CH_Exp6_ZFPtm1a_M14_mask.nii",
        "/content/data_folder/test_mask/20170920CH_Exp09_M12_mask.nii",
        "/content/data_folder/test_mask/20170921CH_Exp09_M24_mask.nii",
        "/content/data_folder/test_mask/20190320CH_Exp4_M18_mask.nii"
    ]

    selected_slices = [16, 17, 18, 19]
    fig, axes = plt.subplots(nrows=len(img_paths), ncols=len(selected_slices), figsize=(4*len(selected_slices), 4*len(img_paths)))

    for row, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        plot_slices(img_path, mask_path, selected_slices, model, model_path, device, row, axes)

    plt.tight_layout()
    plt.show()
