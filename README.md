````markdown
# 🔬 Hướng dẫn Predict Ảnh Tế Bào sử dụng nnUNetv2 (Dataset777_Final)

Đây là hướng dẫn chi tiết để thiết lập môi trường, chuẩn bị dữ liệu và chạy quá trình **Predict (Dự đoán/Phân đoạn)** trên ảnh tế bào RGB bằng mô hình nnUNetv2 đã được huấn luyện.

---

## 1. ⚙️ Cài đặt Môi trường và Thư viện

### 1.1. Thiết lập Môi trường Conda (Khuyến nghị)

Nên sử dụng **Conda** để tạo một môi trường ảo riêng biệt cho dự án nhằm tránh xung đột thư viện.

```bash
conda create -n lab python=3.9  # Chọn phiên bản Python phù hợp
conda activate lab
````

### 1.2. Cài đặt Thư viện Python

Sử dụng tệp `requirements.txt` (đã được tạo ra trước đó) để cài đặt tất cả các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

*(Nếu bạn chưa có `requirements.txt`, hãy tạo nó với các thư viện cần thiết như `nnunetv2`, `numpy`, `opencv-python`, v.v.)*

-----

## 2\. 🗃️ Cập nhật Mô hình nnUNetv2 đã Huấn luyện

### 2.1. Xác định Thư mục nnUNetv2

Tìm đường dẫn thư mục mà nnUNetv2 đã được cài đặt. Ví dụ:

```
C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2
```

### 2.2. Tải và Thay thế Dữ liệu

1.  **Tải xuống** 4 thư mục trong liên kết Google Drive sau:
    > [https://drive.google.com/drive/folders/1A0W63pZyDdFNp6kFE7pRPrF6mju742bU?usp=sharing](https://drive.google.com/drive/folders/1A0W63pZyDdFNp6kFE7pRPrF6mju742bU?usp=sharing)
2.  **Copy** cả 4 thư mục này.
3.  **Paste** và **thay thế** (replace all) chúng vào thư mục nnUNetv2 đã xác định ở bước 2.1.

-----

## 3\. 🖼️ Chuẩn bị Dữ liệu Input

Input của quá trình Predict là **Ảnh tế bào RGB**.

### 3.1. Chuyển đổi RGB sang Grayscale (B1)

Sử dụng script `convert_togray.py` để chuyển đổi ảnh RGB đầu vào thành ảnh Grayscale.

### 3.2. Chuyển đổi Grayscale sang Định dạng nii.gz (B2)

Sử dụng script `nii.py` để chuyển đổi các ảnh Grayscale thành định dạng chuẩn cho nnUNetv2 là **`.nii.gz`**.

### 3.3. Tổ chức Thư mục Input (B3 & B4)

Sắp xếp các file ảnh đã chuẩn bị vào các thư mục sau:

| Thư mục | Định dạng File | Mục đích |
| :--- | :--- | :--- |
| `...\nnunetv2\nnUNet_raw\Dataset777_Final\imagesTs` | **`.nii.gz`** | Input chính cho nnUNetv2 Predict |
| `...\nnunetv2\nnUNet_raw\Dataset777_Final\imagesTs_PNG` | **Ảnh gốc (RGB/PNG)** | Dùng cho bước hậu xử lý cuối cùng (combined\_new.py) |

> **⚠️ QUAN TRỌNG:**
>
> **Trước mỗi lần chạy Predict mới**, bạn **PHẢI** dọn dẹp (xóa hết file) trong 3 thư mục sau để đảm bảo kết quả mới nhất:
>
> 1.  `imagesTs`
> 2.  `imagesTs_PNG`
> 3.  `nnUNet_results\Dataset777_Final\nnUNetTrainer__nnUNetPlans__2d\imagesTs_results`

-----

## 4\. 🚀 Tiến hành Predict

### 4.1. Di chuyển và Thiết lập Environment

1.  Mở **Terminal** hoặc **Anaconda Prompt** (đã activate môi trường `lab`).

2.  Di chuyển vào thư mục cài đặt nnUNetv2:

    ```bash
    cd C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2
    ```

3.  **Thiết lập các biến môi trường (Environment Variables)** cho nnUNet:

    ```bash
    set nnUNet_raw=C:/Users/AlarmTran/anaconda3/envs/lab/Lib/site-packages/nnunetv2/nnUNet_raw
    set nnUNet_preprocessed=C:/Users/AlarmTran/anaconda3/envs/lab/Lib/site-packages/nnunetv2/nnUNet_preprocessed
    set nnUNet_results=C:/Users/AlarmTran/anaconda3/envs/lab/Lib/site-packages/nnunetv2/nnUNet_results
    ```

    *(Sử dụng `/` thay vì `\` trong đường dẫn của biến môi trường là phương pháp tốt)*

### 4.2. Chạy Lệnh Predict (B7)

Thực hiện dự đoán bằng lệnh sau:

```bash
nnUNetv2_predict -d 777 -f all -c 2d -i nnUNet_raw/Dataset777_Final/imagesTs -o nnUNet_results/Dataset777_Final/nnUNetTrainer__nnUNetPlans__2d/imagesTs_results --save_probabilities
```

  * `-d 777`: Chỉ định ID của Dataset (`Dataset777_Final`).
  * `-f all`: Sử dụng tất cả các fold (mô hình đã được huấn luyện).
  * `-c 2d`: Chỉ định cấu hình 2D.
  * `--save_probabilities`: Lưu trữ các bản đồ xác suất (probabilities).

**Kết quả Mask dự đoán** sẽ được lưu trong:
`nnUNet_results/Dataset777_Final/nnUNetTrainer__nnUNetPlans__2d/imagesTs_results`

-----

## 5\. 🎨 Hậu xử lý và Kết hợp Kết quả

Sử dụng script `combined_new.py` để kết hợp ảnh Mask dự đoán (`.nii.gz`) với ảnh gốc (PNG/RGB) để tạo ra kết quả cuối cùng.

### 5.1. Chỉnh sửa `combined_new.py` (B8 & B9)

Mở file `combined_new.py` và cập nhật các đường dẫn sau:

1.  **Đường dẫn ảnh Mask (Sau khi Predict):**

    ```python
    nnunet_dir = r"C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2\nnUNet_results\Dataset777_Final\nnUNetTrainer__nnUNetPlans__2d\imagesTs_results" # Đường dẫn folder Mask
    ```

2.  **Đường dẫn ảnh Input Gốc (RGB):**

    ```python
    input_dir = r"C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2\nnUNet_raw\Dataset777_Final\imagesTs_PNG" # Đường dẫn folder Ảnh gốc
    ```

3.  **Đường dẫn Output (Tùy chọn):**
    Bạn có thể thay đổi biến `output_dir` trong file `combined_new.py` nếu muốn lưu kết quả ở một thư mục khác.

### 5.2. Chạy Hậu xử lý (B10)

Chạy file script cuối cùng:

```bash
python combined_new.py
```

*(Chờ quá trình chạy hoàn tất để nhận được ảnh kết quả cuối cùng.)*

```
```