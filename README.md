# CBAM_Tumor_Classification

**Setup and Run Instructions**

Prerequisites: User has access to NVIDIA GPUs and CUDA Computing Toolkit on either a local or remote machine.

1. Download repository
2. Run preprocessing ipynb file
3. Run training file
4. Run evaluation.py file

The dataset is already downloaded and stored as a zip file in this repository. However, the links for the dataset are also given below. The preprocessing file will use the data from the zip files to create a new augmented dataset for the training data (testing data remains the same). These augmentations are random and specified in the preprocessing file. The pt files for these datasets are large (around 10 to 15 GB).

**Dataset Link**
* https://huggingface.co/datasets/sartajbhuvaji/Brain-Tumor-Classification

**Research Paper Model Links**
* https://drive.google.com/file/d/1kAVe6Upo_81-NTZv53JtjcZAyUSs5PoL/view?usp=drive_link
* https://drive.google.com/file/d/1dEbJyeBgmKNZPb8dDVnCWntMyg_zZjBu/view?usp=drive_link
