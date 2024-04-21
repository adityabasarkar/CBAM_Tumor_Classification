# CBAM_Tumor_Classification

**Setup and Run Instructions**

Prerequisites: User has access to NVIDIA GPUs and CUDA Computing Toolkit on either a local or remote machine.

1. Download or clone repository
2. Go to the link and download the Training.zip and Testing.zip dataset files.
3. Place the two zip files in the "Classifier" folder 
4. Run preprocessing ipynb file
5. Run training file
6. Run evaluation.py file

The links for the dataset are given below. The preprocessing file will use the data from the zip files to create a new augmented dataset for the training data (testing data remains the same). These augmentations are random and specified in the preprocessing file. The pt files for these datasets are large (around 10 to 15 GB).

**Dataset Link**
* https://huggingface.co/datasets/sartajbhuvaji/Brain-Tumor-Classification

The following are the links to models that were used in the research paper. Download the models and place them in /Classifier/content/models after creating the models folde.

**Research Paper Model Links**
* https://drive.google.com/file/d/1kAVe6Upo_81-NTZv53JtjcZAyUSs5PoL/view?usp=drive_link
* https://drive.google.com/file/d/1dEbJyeBgmKNZPb8dDVnCWntMyg_zZjBu/view?usp=drive_link
