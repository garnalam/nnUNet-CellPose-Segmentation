import os
from skimage import io
import SimpleITK as sitk
import numpy as np

def convert_2d_images_in_folder_to_nifti(input_folder: str, output_folder: str, spacing=(999, 1, 1),
                                         transform=None, is_seg: bool = False) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        input_filepath = os.path.join(input_folder, filename)

        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            img = io.imread(input_filepath)
            if transform is not None:
                img = transform(img)

            if len(img.shape) == 2:  # 2D image with no color channels
                img = img[None, None]  # add dimensions
                
            else:
                assert len(img.shape) == 3, "image should be 3D with color channel last but has shape %s" % str(img.shape)
                # We assume that the color channel is the last dimension. Transpose it to be in first
                img = img.transpose((2, 0, 1))
                # Add third dimension
                img = img[:, None]

            # Image is now (c, x, x, z) where x=1 since it's 2D
            if is_seg:
                assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

            for j, i in enumerate(img):
                if is_seg:
                    i = i.astype(np.uint32)

                itk_img = sitk.GetImageFromArray(i)
                itk_img.SetSpacing(list(spacing)[::-1])

                output_filename_truncated = os.path.join(output_folder, os.path.splitext(filename)[0])
                if not is_seg:
                    sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
                else:
                    sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")

# Example usage
input_folder = r"E:\DIC-C2DH-HeLa\DIC-C2DH-HeLa\GRAY"
output_folder = r"E:\DIC-C2DH-HeLa\DIC-C2DH-HeLa\GRAY_NII"
convert_2d_images_in_folder_to_nifti(input_folder, output_folder,is_seg=False)
