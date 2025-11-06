import os
from skimage import io, color, img_as_ubyte
import cv2

def convert_rgb_to_gray(input_folder: str, output_folder: str) -> None:
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        if filename.endswith(('.png', '.jpg', '.jpeg','.tif')):
            image = cv2.imread(input_path)
            
            gray_image = color.rgb2gray(image)
            
            gray_image_8bit = img_as_ubyte(gray_image)
            
            output_path = os.path.join(output_folder, filename)
            
            io.imsave(output_path, gray_image_8bit)

input_folder = r"E:\DIC-C2DH-HeLa\DIC-C2DH-HeLa\01"
output_folder = r"E:\DIC-C2DH-HeLa\DIC-C2DH-HeLa\GRAY"
convert_rgb_to_gray(input_folder, output_folder)
