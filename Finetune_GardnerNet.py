# %% [markdown]
# # Library

# %%
import torch
import tqdm
from tqdm import trange, tqdm
import os

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import cv2
import numpy
import numpy as np

import PIL
from PIL import Image, ImageEnhance

import random
import pandas as pd
import timm

from sklearn.metrics import roc_curve, auc
from sklearn.utils import class_weight
import copy

import torch.optim as optim

from collections import OrderedDict

# %% [markdown]
# # Dataset

# %%
def get_mean_std(src_img_dir, all_img_names):
    
    tf_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                ])

    #all_img_names = [f for f in os.listdir(src_img_dir) if os.path.isfile(os.path.join(src_img_dir, f))]
    
    #VAR[X] = E[X**2] - E[X]**2
    channel_sum, channel_squared_sum, num_batches = 0, 0, 0

    for idx in trange(len(all_img_names)):
        #data = data.repeat(1, 3, 1, 1)
        #data shape: batchSize, 1, H, W

        cur_loaded_img = cv2.imdecode(np.fromfile(src_img_dir+all_img_names[idx], dtype= np.uint8), cv2.IMREAD_GRAYSCALE)
        cur_pil_img = Image.fromarray(cur_loaded_img)
        tensor_img = tf_to_tensor(cur_pil_img).unsqueeze(0)

        channel_sum += torch.mean( tensor_img, dim=[0,2,3])
        channel_squared_sum += torch.mean(tensor_img**2, dim=[0,2,3])
        num_batches = num_batches + 1

    mean = channel_sum / num_batches
    std = (channel_squared_sum/num_batches - mean**2)**0.5

    return mean.item(), std.item()

# %%
class BlastocystFocalDataset(Dataset):
    def __init__(self, df_excel, src_img_dir, resize, dataset_mean, dataset_std, num_multifocus_imgs, is_train):
      super(BlastocystFocalDataset, self).__init__()

      self.df_excel = df_excel
      self.resize = resize
      self.MEAN = dataset_mean
      self.STD  = dataset_std

      self.is_train = is_train

      self.list_of_img_list = []
      self.expansion_labels_list = []
      self.icm_labels_list = []
      self.te_labels_list = []

      focus_name_columns = self.df_excel.columns.to_list()[:num_multifocus_imgs]
      label_columns = self.df_excel.columns.to_list()[num_multifocus_imgs:]
      
      min_expansion_label = self.df_excel[ label_columns[0] ].min()
      min_icm_label = self.df_excel[ label_columns[1] ].min()
      min_te_label = self.df_excel[ label_columns[2] ].min()
      

      for idx in trange( len(self.df_excel)  ):

        cur_img_list = []
        for focus_name in focus_name_columns:
            
            cur_img_name = df_excel.iloc[idx][focus_name]

            cur_img_full_path = src_img_dir + cur_img_name
            cur_loaded_img = cv2.imdecode(np.fromfile(cur_img_full_path, dtype= np.uint8), cv2.IMREAD_GRAYSCALE)

            if self.resize != 500:
                cur_loaded_img = cv2.resize( cur_loaded_img, (self.resize,self.resize), cv2.INTER_AREA  )

            cur_img_list.append(cur_loaded_img)
        self.list_of_img_list.append(cur_img_list)

        cur_expansion_label  = int( self.df_excel.iloc[idx][ label_columns[0] ] - min_expansion_label   )
        cur_icm_label = int( self.df_excel.iloc[idx][ label_columns[1] ] - min_icm_label  )     # 1(A), 2(B), 3(C)
        cur_te_label  = int( self.df_excel.iloc[idx][ label_columns[2] ] - min_te_label   )    # 1(A), 2(B), 3(C)

        self.expansion_labels_list.append(  cur_expansion_label    )
        self.icm_labels_list.append( cur_icm_label   )
        self.te_labels_list.append(  cur_te_label    )


    def __len__(self):

        return len(self.df_excel.index)


    def denormalize(self, x_hat):

        img_mean = self.MEAN
        img_std = self.STD
        x = x_hat * img_std + img_mean
        return x


    def __getitem__(self, idx):

        # ds label
        expansion_label_outcome_list = self.expansion_labels_list[idx]
        expansion_label_outcome_array = np.array([expansion_label_outcome_list])
        expansion_label_outcome_float_array = expansion_label_outcome_array.astype('float')
        expansion_label_outcome_tensor = torch.from_numpy(expansion_label_outcome_float_array)
        expansion_label_outcome_long_tensor = expansion_label_outcome_tensor.to(torch.long)
        # icm label
        icm_label_outcome_list = self.icm_labels_list[idx]
        icm_label_outcome_array = np.array([icm_label_outcome_list])
        icm_label_outcome_float_array = icm_label_outcome_array.astype('float')
        icm_label_outcome_tensor = torch.from_numpy(icm_label_outcome_float_array)
        icm_label_outcome_long_tensor = icm_label_outcome_tensor.to(torch.long)
        # te label
        te_label_outcome_list = self.te_labels_list[idx]
        te_label_outcome_array = np.array([te_label_outcome_list])
        te_label_outcome_float_array = te_label_outcome_array.astype('float')
        te_label_outcome_tensor = torch.from_numpy(te_label_outcome_float_array)
        te_label_outcome_long_tensor = te_label_outcome_tensor.to(torch.long)

        rotation_degree = random.randrange(0, 350, 10)
        brightness_factor = random.randrange(7, 13, 1); brightness_factor = brightness_factor / 10.0
        tf_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                ])

        opencv_img_list_raw = self.list_of_img_list[idx]
        tensor_img_list = []

        for img_idx in range( len(opencv_img_list_raw)  ):

            cur_opencv_img = opencv_img_list_raw[img_idx]
            cur_pil_img = Image.fromarray(cur_opencv_img)

            if self.is_train:
                # rotation
                cur_pil_img = cur_pil_img.rotate(angle= rotation_degree, fillcolor= int(self.MEAN*255)) # rotate
                # adjust brightness
                enhancer = ImageEnhance.Brightness(cur_pil_img)
                cur_pil_img = enhancer.enhance(brightness_factor)

            tensor_img = tf_to_tensor(cur_pil_img)
            tensor_img = (tensor_img - self.MEAN) / self.STD
            tensor_img_list.append(tensor_img)

        final_tensor_img = torch.cat( tensor_img_list, dim= 0  )

        return final_tensor_img, expansion_label_outcome_long_tensor, icm_label_outcome_long_tensor, te_label_outcome_long_tensor

# %% [markdown]
# # Model

# %%
class GardnerNet(nn.Module):
    def __init__(self, num_of_multifocus_images, num_expansion_classes):
        super(GardnerNet, self).__init__()

        self.model_ft = timm.create_model( 'resnet152', pretrained= False, in_chans= 1, num_classes= 2)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Identity()

        self.number_of_multifocus_images = num_of_multifocus_images

        self.ds_layer = nn.Linear(self.num_ftrs*num_of_multifocus_images, num_expansion_classes)   # expansion: 3, 4, 5, 6
        self.icm_layer = nn.Linear(self.num_ftrs*num_of_multifocus_images, 3)  # (1)(A)(good), (2)(B)(fair), (3)(C)(poor)
        self.te_layer = nn.Linear(self.num_ftrs*num_of_multifocus_images, 3)   # (1)(A)(good), (2)(B)(fair), (3)(C)(poor)

        self.softmax_op = torch.nn.Softmax(dim=1)

    def forward(self, x): # x.shape: batch, number_of_multifocus_images, height, width
        features = []
        for idx in range( x.shape[1] ):
            features.append( self.model_ft( x[:,idx,:].unsqueeze(1)  )  )

        concatenated_features = torch.cat( features, dim=1  )

        expansion_out  = self.ds_layer(concatenated_features)
        icm_out = self.icm_layer(concatenated_features)
        te_out  = self.te_layer(concatenated_features)

        return expansion_out, icm_out, te_out

# %% [markdown]
# # Train, Val, Test functions

# %%
def train_model( model, device, dataloader_source, criterion_ds, criterion_icm, criterion_te, optimizer):

  model.train()

  len_dataloader   = len(dataloader_source)

  with tqdm(total= len_dataloader) as pbar_batch:
    for x, expansion_labels, icm_labels, te_labels in dataloader_source:
        x = x.to(device)

        expansion_labels = torch.squeeze(expansion_labels, 1)
        expansion_labels = expansion_labels.to(device)

        icm_labels = torch.squeeze(icm_labels, 1)
        icm_labels = icm_labels.to(device)

        te_labels = torch.squeeze(te_labels, 1)
        te_labels = te_labels.to(device)

        optimizer.zero_grad()

        expansion_out, icm_out, te_out = model(x)

        loss_ds =  criterion_ds(expansion_out,  expansion_labels)
        loss_icm =  criterion_icm(icm_out,  icm_labels)
        loss_te =  criterion_te(te_out,  te_labels)
        loss = (loss_ds + loss_icm + loss_te) / 3.0

        loss.backward()

        optimizer.step()

        pbar_batch.update(1)

# %%
def ROC_Curve(y_test, y_score):
  fpr = dict()
  tpr = dict()
  n_classes = y_test.shape[1]

  roc_auc = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # macro-average AUC
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
  # Finally average it and compute AUC
  mean_tpr /= n_classes
  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  return roc_auc["macro"]

# %%
def test_model_confusion_matrix(model, dataloaders, device, num_expansion_classes):

  model.eval()

  # expansion
  expansion_running_corrects = 0
  if 3 == num_expansion_classes:
    expansion_confusion_matrix = np.zeros( (3, 3) ) # 
    expansion_look_up_table = [ [1,0,0], [0,1,0], [0,0,1] ]
  elif 4 == num_expansion_classes:
    expansion_confusion_matrix = np.zeros( (4, 4) ) # 
    expansion_look_up_table = [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ]
  expansion_truth = []; expansion_scores = []

  # ICM
  icm_running_corrects = 0
  icm_confusion_matrix = np.zeros( (3, 3) )
  icm_truth = []; icm_scores = []

  # TE
  te_running_corrects = 0
  te_confusion_matrix = np.zeros( (3, 3) )
  te_truth = []; te_scores = []

  softmax_op = nn.Softmax(dim=1)

  check_log = 0

  tqdm_total = len(dataloaders)
  with tqdm(total= tqdm_total) as bar:
    with torch.no_grad():
      for x, expansion_labels, icm_labels, te_labels in dataloaders:

        x = x.to(device)

        expansion_labels = torch.squeeze(expansion_labels, 1)
        expansion_labels = expansion_labels.to(device)

        icm_labels = torch.squeeze(icm_labels, 1)
        icm_labels = icm_labels.to(device)

        te_labels = torch.squeeze(te_labels, 1)
        te_labels = te_labels.to(device)

        expansion_out, icm_out, te_out = model(x)

        # metric related to DS
        _, expansion_preds = torch.max( expansion_out, 1)
        expansion_running_corrects += torch.sum(expansion_preds == expansion_labels.data)
        expansion_predict_numpy = expansion_preds.cpu().numpy()
        rows = expansion_predict_numpy.shape[0]
        for i in range(rows):
          expansion_label_val = int(expansion_labels[i])
          expansion_predict_val = int(expansion_predict_numpy[i])
          expansion_confusion_matrix[expansion_label_val][expansion_predict_val] += 1
        # save scores & labels
        expansion_out = softmax_op( expansion_out  )
        for bIdx in range(expansion_out.shape[0]):
          expansion_truth.append( expansion_look_up_table[ expansion_labels[bIdx].data.cpu().numpy() ] ); expansion_scores.append(expansion_out[bIdx,:].data.cpu().numpy())

        # metric related to ICM
        _, icm_preds = torch.max( icm_out, 1)
        icm_running_corrects += torch.sum( icm_preds == icm_labels.data)
        icm_predict_numpy = icm_preds.cpu().numpy()
        rows = icm_predict_numpy.shape[0]
        for i in range(rows):
          icm_label_val = int(icm_labels[i])
          icm_predict_val = int(icm_predict_numpy[i])
          icm_confusion_matrix[icm_label_val][icm_predict_val] += 1
      #save scores & labels
        icm_out = softmax_op( icm_out  )
        for bIdx in range(icm_out.shape[0]):
          icm_look_up_table = [ [1,0,0], [0,1,0], [0,0,1] ]
          icm_truth.append( icm_look_up_table[ icm_labels[bIdx].data.cpu().numpy() ] ); icm_scores.append(icm_out[bIdx,:].data.cpu().numpy())

        # metric related to TE
        _, te_preds = torch.max( te_out, 1)
        te_running_corrects += torch.sum( te_preds == te_labels.data)
        te_predict_numpy = te_preds.cpu().numpy()
        rows = te_predict_numpy.shape[0]
        for i in range(rows):
          te_label_val = int(te_labels[i])
          te_predict_val = int(te_predict_numpy[i])
          te_confusion_matrix[te_label_val][te_predict_val] += 1
        # save scores & labels
        te_out = softmax_op( te_out  )
        for bIdx in range(te_out.shape[0]):
          te_look_up_table = [ [1,0,0], [0,1,0], [0,0,1] ]
          te_truth.append( te_look_up_table[ te_labels[bIdx].data.cpu().numpy() ] ); te_scores.append(te_out[bIdx,:].data.cpu().numpy())

        bar.update(1)

  expansion_acc = expansion_running_corrects.double() / len(dataloaders.dataset)
  expansion_truth = numpy.array(expansion_truth); expansion_scores = numpy.array(expansion_scores)
  expansion_auc = ROC_Curve( expansion_truth, expansion_scores  )

  icm_acc = icm_running_corrects.double() / len(dataloaders.dataset)
  icm_truth = numpy.array(icm_truth); icm_scores = numpy.array(icm_scores)
  icm_auc = ROC_Curve( icm_truth, icm_scores  )

  te_acc = te_running_corrects.double() / len(dataloaders.dataset)
  te_truth = numpy.array(te_truth); te_scores = numpy.array(te_scores)
  te_auc = ROC_Curve( te_truth, te_scores  )

  return expansion_auc, expansion_acc, expansion_confusion_matrix, icm_auc, icm_acc, icm_confusion_matrix, te_auc, te_acc, te_confusion_matrix

# %%
def GetClassBalancedWeights(df, column_name):
  df_copy = copy.deepcopy(df)

  min_val = df_copy[column_name].min()
  df_copy[column_name] = (df_copy[column_name] - min_val)

  keys = np.unique( df_copy[column_name] )
  values = class_weight.compute_class_weight(class_weight='balanced', classes= keys, y= df_copy[column_name].values)
  values = torch.from_numpy(values)
  values = values.to(torch.float32)

  return values

# %%
def train_evaluate(hyper_parameters, df_train, df_val, df_test, 
                   expansion_column_name, icm_column_name, te_column_name, 
                   device, num_expansion_classes, src_img_dir,
                   dataset_mean, dataset_std, num_multifocus_imgs, 
                   pretrained_model, model_save_dir):

  # weight
  expansion_class_weight = GetClassBalancedWeights(df_train, expansion_column_name)
  expansion_class_weight = expansion_class_weight.to(device)

  icm_class_weight = GetClassBalancedWeights(df_train, icm_column_name)
  icm_class_weight = icm_class_weight.to(device)

  te_class_weight = GetClassBalancedWeights(df_train, te_column_name)
  te_class_weight = te_class_weight.to(device)


  # hyper-parameters
  image_size = hyper_parameters['image_size']
  cur_batch_size = hyper_parameters['batch_size']
  learning_rate = hyper_parameters['lr']
  weight_decay = hyper_parameters['weight_decay']

  # dataset
  db_train = BlastocystFocalDataset(df_train, src_img_dir, image_size, dataset_mean, dataset_std, num_multifocus_imgs, is_train= True)
  db_val   = BlastocystFocalDataset(df_val, src_img_dir, image_size, dataset_mean, dataset_std, num_multifocus_imgs, is_train= False)
  db_test  = BlastocystFocalDataset(df_test, src_img_dir, image_size, dataset_mean, dataset_std, num_multifocus_imgs, is_train= False)


  criterion_ds = torch.nn.CrossEntropyLoss(weight= expansion_class_weight)
  criterion_icm = torch.nn.CrossEntropyLoss(weight= icm_class_weight)
  criterion_te = torch.nn.CrossEntropyLoss(weight= te_class_weight)

  # untrained net
  model = GardnerNet(num_multifocus_imgs, num_expansion_classes)

  # use pretrained encoder
  model.model_ft = copy.deepcopy(pretrained_model.model_ft)

  # multiple gpus training: 
  model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
  model.to(device)

  # params to update
  params_to_update = []
  for param in model.parameters():
      if param.requires_grad == True:
          params_to_update.append(param)


  optimizer = optim.AdamW(params_to_update, lr= learning_rate, weight_decay= weight_decay )

  train_loader = torch.utils.data.DataLoader(db_train, batch_size= cur_batch_size, shuffle= True, drop_last=True)
  val_loader = torch.utils.data.DataLoader(db_val, batch_size= 32, shuffle= False)
  test_loader = torch.utils.data.DataLoader(db_test, batch_size= 32, shuffle= False)


  MAX_EPOCH = 300
  max_auc = 0
  decrease_count = 0

  best_model = ''
  PATIENCE_COUNT = 10


  for idx in range(MAX_EPOCH):
    train_model( model, device, train_loader, criterion_ds, criterion_icm, criterion_te, optimizer)
    (expansion_val_auc, expansion_val_acc, expansion_val_conf, 
     icm_val_auc, icm_val_acc, icm_val_conf, 
     te_val_auc, te_val_acc, te_val_conf) = test_model_confusion_matrix( model, val_loader, device, num_expansion_classes)

    val_auc  = (expansion_val_auc + icm_val_auc + te_val_auc) / 3.0
    val_acc  = (expansion_val_acc + icm_val_acc + te_val_acc) / 3.0

    print('\nEpoch: {}, mean_val_auc: {}'.format(idx, val_auc))
    print( 'expansion_val_auc: {}, expansion_val_acc: {}, expansion_val_conf: {}'.format(expansion_val_auc, expansion_val_acc, expansion_val_conf) )
    print( 'icm_val_auc: {}, icm_val_acc: {}, icm_val_conf: {}'.format(icm_val_auc, icm_val_acc, icm_val_conf) )
    print( 'te_val_auc: {}, te_val_acc: {}, te_val_conf: {}'.format(te_val_auc, te_val_acc, te_val_conf) )

    status = ''
    if val_auc > max_auc:
      max_auc = val_auc
      decrease_count = 0
      best_model =   copy.deepcopy(model)
      status = 'success'
    else:
      status = 'failed'
      decrease_count += 1
      if PATIENCE_COUNT == decrease_count:
        break

    print(status)



  (expansion_test_auc, expansion_test_acc, expansion_test_conf, 
   icm_test_auc, icm_test_acc, icm_test_conf, 
   te_test_auc, te_test_acc, te_test_conf)  = test_model_confusion_matrix( best_model, test_loader, device, num_expansion_classes)

  test_auc  = (expansion_test_auc + icm_test_auc + te_test_auc) / 3.0
  print('\nmean_test_auc: {}'.format(test_auc))
  print( 'expansion_test_auc: {}, expansion_test_acc: {}, expansion_test_conf: {}'.format(expansion_test_auc, expansion_test_acc, expansion_test_conf) )
  print( 'icm_test_auc: {}, icm_test_acc: {}, icm_test_conf: {}'.format(icm_test_auc, icm_test_acc, icm_test_conf) )
  print( 'te_test_auc: {}, te_test_acc: {}, te_test_conf: {}'.format(te_test_auc, te_test_acc, te_test_conf) )

  torch.save( best_model.module.state_dict(), model_save_dir + 'final_model.pt' )

# %% [markdown]
# # Prepare Data and Train

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print( device )

# %%
# load excel
df = pd.read_excel( './fine-tune-dataset/df_5_focus.xlsx' )

# %%
df_shuffle = df.sample(frac=1, random_state= 66).reset_index(drop=True)

TRAIN_SIZE = int( len(df_shuffle) * 0.8 )
VAL_SIZE   = int( len(df_shuffle) * 0.10 )

df_train = df_shuffle.iloc[ : TRAIN_SIZE ]
df_val  = df_shuffle.iloc[ TRAIN_SIZE : TRAIN_SIZE+VAL_SIZE ]
df_test      = df_shuffle.iloc[ TRAIN_SIZE+VAL_SIZE : ]

# %%
# mean, stdev of the training dataset

# get dataset mean, std
src_img_dir = './fine-tune-dataset/imgs/'

num_multifocus_imgs = 5
img_name_columns = df_train.columns.tolist()[:num_multifocus_imgs]
all_img_names = df_train[img_name_columns].values.flatten().tolist()

dataset_mean, dataset_std = get_mean_std( src_img_dir, all_img_names )

print( 'dataset mean: {}, stdev: {}'.format(dataset_mean, dataset_std) )

# %%
# pretrained model

pretrained_pt = torch.load( './GardnerNet_Pretrained_1.pt', map_location= device )

pretrained_model = GardnerNet(num_of_multifocus_images= 2, num_expansion_classes= 4)

pretrained_model.load_state_dict( pretrained_pt )

pretrained_model.to(device)

# %%
hyper_parameters= {'batch_size': 9, 
             'lr': 4.73345487439063e-05, 
             'weight_decay': 0.507309243983485, 
             'image_size': 300 }


expansion_column_name = 'DS'
icm_column_name = 'ICM'
te_column_name = 'TE'

num_expansion_classes = 3
src_img_dir = './fine-tune-dataset/imgs/'
num_multifocus_imgs = 5
model_save_dir = './fine-tune-dataset/'

train_evaluate(hyper_parameters, df_train, df_val, df_test, 
                   expansion_column_name, icm_column_name, te_column_name, 
                   device, num_expansion_classes, src_img_dir,
                   dataset_mean, dataset_std, num_multifocus_imgs, 
                   pretrained_model, model_save_dir)


