````markdown
# 🔬 Cell Image Prediction Guide using nnUNetv2 (Dataset777_Final)

This is a detailed guide for setting up the environment, preparing data, and running the **Predict** process (Segmentation) on RGB cell images using the trained nnUNetv2 model.

---

## 1. ⚙️ Environment and Library Setup

### 1.1. Conda Environment Setup (Recommended)

It is highly recommended to use **Conda** to create a separate virtual environment for the project to prevent library conflicts.

```bash
conda create -n lab python=3.9  # Choose the appropriate Python version
conda activate lab
````

### 1.2. Install Python Libraries

Use the `requirements.txt` file (created previously) to install all necessary libraries:

```bash
pip install -r requirements.txt
```

*(If you don't have `requirements.txt`, create it with necessary libraries like `nnunetv2`, `numpy`, `opencv-python`, etc.)*

-----

## 2\. 🗃️ Update Trained nnUNetv2 Model

### 2.1. Identify the nnUNetv2 Folder

Locate the installation path of the nnUNetv2 library. Example:

```
C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2
```

### 2.2. Download and Replace Data

1.  **Download** the 4 folders from the following Google Drive link:
    > [https://drive.google.com/drive/folders/1A0W63pZyDdFNp6kFE7pRPrF6mju742bU?usp=sharing](https://drive.google.com/drive/folders/1A0W63pZyDdFNp6kFE7pRPrF6mju742bU?usp=sharing)
2.  **Copy** all 4 folders.
3.  **Paste** and **replace all** files into the nnUNetv2 installation folder identified in step 2.1.

-----

## 3\. 🖼️ Input Data Preparation

The input for the Prediction process is **RGB Cell Images**.

### 3.1. Convert RGB to Grayscale (Step B1)

Use the `convert_togray.py` script to convert the input RGB images to Grayscale images.

### 3.2. Convert Grayscale to nii.gz Format (Step B2)

Use the `nii.py` script to convert the Grayscale images into the standard nnUNetv2 format: **`.nii.gz`**.

### 3.3. Organize Input Directories (Steps B3 & B4)

Arrange the prepared image files into the following directories:

| Directory | File Format | Purpose |
| :--- | :--- | :--- |
| `...\nnunetv2\nnUNet_raw\Dataset777_Final\imagesTs` | **`.nii.gz`** | Primary input for nnUNetv2 Predict |
| `...\nnunetv2\nnUNet_raw\Dataset777_Final\imagesTs_PNG` | **Original Image (RGB/PNG)** | Used for the final post-processing step (`combined_new.py`) |

> **⚠️ IMPORTANT:**
>
> **Before every new Prediction run**, you **MUST** clean (delete all files) the contents of the following 3 folders to ensure fresh results:
>
> 1.  `imagesTs`
> 2.  `imagesTs_PNG`
> 3.  `nnUNet_results\Dataset777_Final\nnUNetTrainer__nnUNetPlans__2d\imagesTs_results`

-----

## 4\. 🚀 Run Prediction

### 4.1. Navigate and Set Environment Variables

1.  Open the **Terminal** or **Anaconda Prompt** (with the `lab` environment activated).

2.  Navigate to the nnUNetv2 installation folder:

    ```bash
    cd C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2
    ```

3.  **Set the Environment Variables** for nnUNet:

    ```bash
    set nnUNet_raw=C:/Users/AlarmTran/anaconda3/envs/lab/Lib/site-packages/nnunetv2/nnUNet_raw
    set nnUNet_preprocessed=C:/Users/AlarmTran/anaconda3/envs/lab/Lib/site-packages/nnunetv2/nnUNet_preprocessed
    set nnUNet_results=C:/Users/AlarmTran/anaconda3/envs/lab/Lib/site-packages/nnunetv2/nnUNet_results
    ```

    *(Using `/` instead of `\` in environment variable paths is good practice)*

### 4.2. Execute the Predict Command (Step B7)

Run the prediction using the following command:

```bash
nnUNetv2_predict -d 777 -f all -c 2d -i nnUNet_raw/Dataset777_Final/imagesTs -o nnUNet_results/Dataset777_Final/nnUNetTrainer__nnUNetPlans__2d/imagesTs_results --save_probabilities
```

  * `-d 777`: Specifies the Dataset ID (`Dataset777_Final`).
  * `-f all`: Uses all folds (trained models).
  * `-c 2d`: Specifies the 2D configuration.
  * `--save_probabilities`: Saves the probability maps.

**The predicted Mask results** will be saved in:
`nnUNet_results/Dataset777_Final/nnUNetTrainer__nnUNetPlans__2d/imagesTs_results`

-----

## 5\. 🎨 Post-processing and Result Combination

Use the `combined_new.py` script to merge the predicted Mask images (`.nii.gz`) with the original (PNG/RGB) images to produce the final output.

### 5.1. Edit `combined_new.py` (Steps B8 & B9)

Open the `combined_new.py` file and update the following paths:

1.  **Path to Mask Images (After Predict):**

    ```python
    nnunet_dir = r"C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2\nnUNet_results\Dataset777_Final\nnUNetTrainer__nnUNetPlans__2d\imagesTs_results" # Mask nnUNet Folder Path
    ```

2.  **Path to Original Input Images (RGB):**

    ```python
    input_dir = r"C:\Users\AlarmTran\anaconda3\envs\lab\Lib\site-packages\nnunetv2\nnUNet_raw\Dataset777_Final\imagesTs_PNG" # Original Image Folder Path
    ```

3.  **Output Path (Optional):**
    You can change the `output_dir` variable in `combined_new.py` if you wish to save the results to a different directory.

### 5.2. Run Post-processing (Step B10)

Execute the final script:

```bash
python combined_new.py
```