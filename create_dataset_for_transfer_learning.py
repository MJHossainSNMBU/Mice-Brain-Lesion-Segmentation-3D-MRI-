
# Import necessary libraries
import os
import numpy as np
import nibabel as nib
import shutil
np.random.seed(42)

# Define the paths to the image and mask folders
image_folder = "/content/data_folder/train_image"
mask_folder = "/content/data_folder/train_mask"

# Create a list of all mask file names
mask_files = os.listdir(mask_folder)

# Initialize a dictionary to store lesion sizes
lesion_sizes = {}

# Iterate over the mask files
for mask_file in mask_files:
    if mask_file.endswith("_mask.nii"):
        # Extract the corresponding image file name
        image_file = mask_file[:-9] + ".nii"

        # Load the image file and its header
        image_path = os.path.join(image_folder, image_file)
        image_header = nib.load(image_path).header

        # Obtain the voxel size from the image header
        voxel_size = np.prod(image_header.get_zooms())

        # Load the mask file
        mask_path = os.path.join(mask_folder, mask_file)
        mask_data = nib.load(mask_path).get_fdata()

        # Calculate the size of the lesion in millimeters
        lesion_size = np.sum(mask_data) * voxel_size

        # Store the lesion size in the dictionary
        lesion_sizes[image_file] = lesion_size

# Convert the lesion sizes to an array
lesion_sizes_array = np.array(list(lesion_sizes.values()))

# Generate bins for the Gaussian distribution based on lesion sizes
bins = np.linspace(np.min(lesion_sizes_array), np.max(lesion_sizes_array), 9)  # Create 8 bins
class_labels = np.digitize(lesion_sizes_array, bins)

# Initialize a dictionary to store the selected images
selected_images = {}

# Select 5 images from each class based on the Gaussian distribution
for class_label in range(1, 9):
    class_indices = np.where(class_labels == class_label)[0]
    if len(class_indices) > 0:
        selected_indices = np.random.choice(class_indices, min(2, len(class_indices)), replace=False)  # Choose 5 or less if not available
        selected_images_from_class = [list(lesion_sizes.keys())[i] for i in selected_indices]
        selected_images.update({img: class_label for img in selected_images_from_class})

# If less than 40 images were selected, select additional images randomly
while len(selected_images) < 16:
    for class_label in range(1, 9):
        class_indices = np.where(class_labels == class_label)[0]
        leftover_indices = [idx for idx in class_indices if list(lesion_sizes.keys())[idx] not in selected_images]
        if len(leftover_indices) > 0:
            selected_index = np.random.choice(leftover_indices)
            selected_image = list(lesion_sizes.keys())[selected_index]
            selected_images[selected_image] = class_label
            if len(selected_images) >= 16:
                break

# Define the paths to the output folders
output_image_folder = "/content/data_folder/selected_images16"
output_mask_folder = "/content/data_folder/selected_masks16"

# Create the output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Copy the selected images and masks to the output folders
for selected_image in selected_images:
    # Copy the image
    source_image_path = os.path.join(image_folder, selected_image)
    destination_image_path = os.path.join(output_image_folder, selected_image)
    shutil.copy2(source_image_path, destination_image_path)

    # Copy the mask
    mask_name = selected_image[:-4] + "_mask.nii"
    source_mask_path = os.path.join(mask_folder, mask_name)
    destination_mask_path = os.path.join(output_mask_folder, mask_name)
    shutil.copy2(source_mask_path, destination_mask_path)

    # Print the information
    print("Image:", selected_image)
    print("Mask:", mask_name)
    print("Class:", selected_images[selected_image])
    print("Lesion Size (mm^3):", lesion_sizes[selected_image])
    print()

print("Selected images and masks have been copied to separate folders.")

# Verify the number of files in the selected dataset
folder_path = '/content/data_folder/selected_images16'
file_list = os.listdir(folder_path)
num_files = len(file_list)
print("Number of files in Train Image folder:", num_files)

folder_path = '/content/data_folder/selected_masks16'
file_list = os.listdir(folder_path)
num_files = len(file_list)
print("Number of files in train mask folder:", num_files)
