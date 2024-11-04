# GardnerNet
**Source code for our manuscript entitled _' GardnerNet: an interpretable deep learning model for quantitative human blastocyst evaluation— a retrospective model  development and validation study.'_**

**Dowload the pretrained model at https://drive.google.com/file/d/15rJip2UT_fRoXl-P9J1_5ZVR9W7vBa6Z/view?usp=sharing**

# Required library:
1. Pytorch with cuda (>2.0): https://pytorch.org/
2. OpenCV-Python (>3.0): https://github.com/opencv/opencv-python
3. Pytorch Image Models: https://github.com/huggingface/pytorch-image-models
4. Python Imaging Library: https://github.com/python-pillow/Pillow
5. Progress Bar for Python: https://github.com/tqdm/tqdm
6. Scikit-learn: https://scikit-learn.org/stable/install.html#installation-instructions

# Test pretrained GardnerNet
Use **_Test_GardnerNet.py_** to test pretrained GardnerNet on example images of blastocysts in Figures 1, 3, and 4 in the manuscript.


# Fine-tune GardnerNet on your own dataset
Use **_Finetune_GardnerNet.py_** to fine-tune GradnerNet on your own dataset.
1. Download pretrained GardnerNet model from https://drive.google.com/file/d/15rJip2UT_fRoXl-P9J1_5ZVR9W7vBa6Z/view?usp=sharing
2. Put all blastocyst images in _./fine-tune-dataset/imgs/_
3. Creat an Excel file containg multi-focus image names and lables of expansion, ICM, and TE (see ./fine-tune-dataset/df_5_focus.xlsx).
   For example, if each blastocyst has five focal planes, the column name and order of the Excel file should be:

   **focus_1_name	focus_2_name	focus_3_name	focus_4_name	focus_5_name	Expansion	ICM	TE**.

   'Expansion' labels (descriptions): 3(full blastocyst), 4 (expanded), 5 (hatching), 6 (hatched).

   'ICM/TE' labels (descriptions): 1 (A, or good), 2 (B or fair), 3 (C or poor)
4. Modify parameters such as num_expansion_classes, num_multifocus_imgs in **_Finetune_GardnerNet.py_** to adapt to your own dataset.
5. Run **_Finetune_GardnerNet.py_**, the validation metrics (e.g., AUC, Acc, confusion matrix) will be shown after each epoch, final model
   will be saved as './fine-tune-dataset/final_model.pt'.
 
