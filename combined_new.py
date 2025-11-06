import numpy as np
import cv2
import os
from cellpose import models
import matplotlib.pyplot as plt
import json

def apply_nnunet_mask_to_image(original_image, nnunet_mask):
    """
    Áp dụng mask nnUNet lên ảnh gốc, giữ lại vùng mask > 0, bỏ vùng còn lại.
    """
    nnunet_binary = (nnunet_mask > 0).astype(np.uint8)
    masked_image = original_image.copy()
    masked_image[nnunet_binary == 0] = 0
    return masked_image

def run_cellpose_on_masked_image(masked_image, model_type='cyto3', diameter=None):
    """
    Chạy Cellpose trên ảnh đã được lọc bởi nnUNet mask.
    """
    model = models.Cellpose(model_type=model_type)
    masks, flows, styles, diams = model.eval(masked_image, diameter=diameter, channels=[0, 0])
    return masks

def visualize_and_save(original_image, nnunet_mask, masked_image, cellpose_mask, output_dir, base_name):
    """
    Trực quan hóa và lưu kết quả, bao gồm mask 16-bit, 8-bit, ảnh gốc với contour và export contours ra JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Hiển thị kết quả
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(142)
    plt.imshow(nnunet_mask > 0, cmap='viridis')
    plt.title('nnUNet Mask')
    plt.axis('off')
    
    plt.subplot(143)
    plt.imshow(masked_image)
    plt.title('Masked Image')
    plt.axis('off')
    
    plt.subplot(144)
    plt.imshow(cellpose_mask, cmap='viridis')
    plt.title('Cellpose on Masked Image')
    plt.axis('off')
    
    plt.tight_layout()
    vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Lưu mask Cellpose 16-bit
    mask_16bit_path = os.path.join(output_dir, f"{base_name}_cellpose_mask_16bit.png")
    cv2.imwrite(mask_16bit_path, cellpose_mask.astype(np.uint16))
    
    # Lưu mask Cellpose 8-bit (chuẩn hóa từ 16-bit về 0-255)
    mask_8bit = cv2.normalize(cellpose_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_8bit_path = os.path.join(output_dir, f"{base_name}_cellpose_mask_8bit.png")
    cv2.imwrite(mask_8bit_path, mask_8bit)
    
    # Tạo danh sách contours để export JSON
    contours_data = []
    unique_labels = np.unique(mask_8bit)
    unique_labels = unique_labels[unique_labels > 0]  # Loại bỏ nền (0)
    
    # Vẽ contour và thu thập dữ liệu cho JSON
    original_with_contours = original_image.copy()
    for idx, label in enumerate(unique_labels):
        binary_mask = (mask_8bit == label).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Vẽ contour lên ảnh gốc
        cv2.drawContours(original_with_contours, contours, -1, (0, 255, 0), 1)
        
        # Chuẩn bị dữ liệu contour cho JSON
        for contour in contours:
            contour_points = contour.squeeze().tolist()
            # Kiểm tra và sửa định dạng contour_points
            if isinstance(contour_points, int):  # Nếu là số nguyên đơn lẻ
                continue  # Bỏ qua contour không hợp lệ
            elif isinstance(contour_points, list) and contour_points:  # Nếu là danh sách
                if not isinstance(contour_points[0], list):  # Nếu là [x1, y1, ...]
                    contour_points = [[contour_points[i], contour_points[i+1]] for i in range(0, len(contour_points), 2)]
                # contour_points giờ là [[x1, y1], [x2, y2], ...]
            else:
                continue  # Bỏ qua nếu không hợp lệ
            contours_data.append({
                "id": idx,
                "points": contour_points  # [[x1, y1], [x2, y2], ...]
            })
    
    # Lưu ảnh gốc với contour
    contour_image_path = os.path.join(output_dir, f"{base_name}_original_with_contours.png")
    cv2.imwrite(contour_image_path, cv2.cvtColor(original_with_contours, cv2.COLOR_RGB2BGR))
    
    # Export contours ra file JSON
    json_path = os.path.join(output_dir, f"{base_name}_contours.json")
    with open(json_path, 'w') as f:
        json.dump(contours_data, f, indent=4)

def main():
    # Đường dẫn thư mục
    input_dir = r"C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2\nnUNet_raw\Dataset777_Final\imagesTs_PNG"  # Ảnh gốc
    nnunet_dir = r"C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2\nnUNet_results\Dataset777_Final\nnUNetTrainer__nnUNetPlans__2d\imagesTs_results\Mask_Results"  # Mask nnUNet
    output_dir = r"C:\Users\AlarmTran\Desktop\Data\Data\cells-seg\combined_results"  # Kết quả
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách file ảnh
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.tif'))])
    
    for img_file in image_files:
        print(f"Processing {img_file}...")
        
        # Đường dẫn file
        img_path = os.path.join(input_dir, img_file)
        base_name = img_file.replace('.jpg', '')  # Tên file không đuôi
        nnunet_path = os.path.join(nnunet_dir, f"{base_name}.dcm.png")  # Giả sử tên file nnUNet
        
        # Đọc ảnh gốc và mask nnUNet
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"Error loading original image: {img_path}")
            continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        nnunet_mask = cv2.imread(nnunet_path, cv2.IMREAD_GRAYSCALE)
        if nnunet_mask is None:
            print(f"Error loading nnUNet mask: {nnunet_path}")
            continue
        
        # Áp mask nnUNet lên ảnh gốc
        masked_image = apply_nnunet_mask_to_image(original_image, nnunet_mask)
        
        # Chạy Cellpose trên ảnh đã lọc
        cellpose_mask = run_cellpose_on_masked_image(masked_image, model_type='cyto3', diameter=None)
        
        # Lưu và trực quan hóa kết quả
        visualize_and_save(original_image, nnunet_mask, masked_image, cellpose_mask, output_dir, base_name)
        
        print(f"Completed {img_file}")

if __name__ == "__main__":
    main()