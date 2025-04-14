# BlastScoringNet
**Source code for our paper entitled _'An interpretable artificial intelligence approach to differentiate between blastocysts with similar or same morphological grades.'_**

**Dowload the pretrained model at https://drive.google.com/file/d/15rJip2UT_fRoXl-P9J1_5ZVR9W7vBa6Z/view?usp=sharing**

# Required library
1. Pytorch with cuda (>2.0): https://pytorch.org/
2. OpenCV-Python (>3.0): https://github.com/opencv/opencv-python
3. Pytorch Image Models: https://github.com/huggingface/pytorch-image-models
4. Python Imaging Library: https://github.com/python-pillow/Pillow
5. Progress Bar for Python: https://github.com/tqdm/tqdm
6. Scikit-learn: https://scikit-learn.org/stable/install.html#installation-instructions
7. Pandas: https://github.com/pandas-dev/pandas

# Test pretrained BlastScoringNet
Use **_Test.py_** to test pretrained BlastScoringNet on example images of blastocysts in Figures 1, 3, and 4 in the paper.


# Pretrained model and hyperparameters for fine-tuning (load backbone encoder and then fine-tune all params)
1. Download pretrained BlastScoringNet model from https://drive.google.com/file/d/15rJip2UT_fRoXl-P9J1_5ZVR9W7vBa6Z/view?usp=sharing
2. Pytorch AdamW hyper_parameters= {'batch_size': 9, 
             'lr': 4.73345487439063e-05, 
             'weight_decay': 0.507309243983485, 
             'image_size': 300 }
 
