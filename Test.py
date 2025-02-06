# %% [markdown]
# # Library

# %%
import torch
import cv2
import torch.nn as nn
import torchvision
from torchvision import transforms

import timm
import numpy as np

import PIL
from PIL import Image

# %% [markdown]
# # Device

# %%
print( torch.__version__ )
print( torch.version.cuda )

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print( device )

# %% [markdown]
# # Model

# %%
# Model definition

class BlastScoringNet(nn.Module):
    def __init__(self, number_of_multifocus_images):
        super(BlastScoringNet, self).__init__()

        self.model_ft = timm.create_model( 'resnet152', pretrained= False, in_chans= 1, num_classes= 2)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Identity()

        self.number_of_multifocus_images = number_of_multifocus_images

        self.ds_layer = nn.Linear(self.num_ftrs*number_of_multifocus_images, 4)   # 3, 4, 5, 6
        self.icm_layer = nn.Linear(self.num_ftrs*number_of_multifocus_images, 3)  # (1)(A)(good), (2)(B)(fair), (3)(C)(poor)
        self.te_layer = nn.Linear(self.num_ftrs*number_of_multifocus_images, 3)   # (1)(A)(good), (2)(B)(fair), (3)(C)(poor)

        self.softmax_op = torch.nn.Softmax(dim=1)

    def forward(self, x): # x.shape: batch, number_of_multifocus_images, height, width
        features = []
        for idx in range( x.shape[1] ):
            features.append( self.model_ft( x[:,idx,:].unsqueeze(1)  )  )

        concatenated_features = torch.cat( features, dim=1  )

        ds_out  = self.ds_layer(concatenated_features)
        icm_out = self.icm_layer(concatenated_features)
        te_out  = self.te_layer(concatenated_features)

        return ds_out, icm_out, te_out

# %%
# Load model
model_path = './BlastScoringNet_Pretrained_1.pt'
saved_model = torch.load( model_path, map_location= device )

# %%
number_of_multifocus_images = 2
model = BlastScoringNet(number_of_multifocus_images)

model.load_state_dict( saved_model )

model = model.to(device)

model.eval()

# %% [markdown]
# # Process blastocyst images to get expansion degree grade (3-6), ICM score, and TE score

# %%
def Preprocess(src_img_dir, src_img_name_list):
    IMG_MEAN = 0.4800113
    IMG_STD  = 0.073582366
    IMG_SIZE = 300

    tf_to_tensor = transforms.Compose([
                transforms.Resize( (IMG_SIZE, IMG_SIZE) ),
                transforms.ToTensor(),
                ])
    
    tensor_img_list = []
    for idx in range( len(src_img_name_list) ):
        cur_img = cv2.imdecode( np.fromfile( src_img_dir + src_img_name_list[idx], dtype=np.uint8), cv2.IMREAD_GRAYSCALE  )
        pil_img = Image.fromarray(cur_img)
        tensor_img = tf_to_tensor(pil_img)
        tensor_img = (tensor_img - IMG_MEAN) / IMG_STD
        tensor_img = torch.unsqueeze(tensor_img, 0)

        tensor_img_list.append(tensor_img)

    final_tensor_img = torch.cat( tensor_img_list, dim= 1 )

    return final_tensor_img


def Postprocess(ds_out, icm_out, te_out):
    softmax_op = torch.nn.Softmax(dim=1)

    ds_prob  = softmax_op(ds_out)
    icm_prob = softmax_op(icm_out)
    te_prob  = softmax_op(te_out)

    if ds_prob.is_cuda:
        ds_prob_numpy_list  =  ds_prob.detach().cpu().numpy().tolist()
        icm_prob_numpy_list =  icm_prob.detach().cpu().numpy().tolist()
        te_prob_numpy_list  =  te_prob.detach().cpu().numpy().tolist()
    else:
        ds_prob_numpy_list = ds_prob.detach().numpy().tolist()
        icm_prob_numpy_list = icm_prob.detach().numpy().tolist()
        te_prob_numpy_list = te_prob.detach().numpy().tolist()

    expnasion_degree_grade = np.argmax(ds_prob_numpy_list[0]) + 3
    icm_score = 4 - ( 1.0 * icm_prob_numpy_list[0][0] + 2.0 * icm_prob_numpy_list[0][1] + 3.0 * icm_prob_numpy_list[0][2] )
    te_score  = 4 - ( 1.0 * te_prob_numpy_list[0][0]  + 2.0 * te_prob_numpy_list[0][1]  + 3.0 * te_prob_numpy_list[0][2] )

    return expnasion_degree_grade, icm_score, te_score, icm_prob_numpy_list, te_prob_numpy_list


def Process( src_img_dir, src_img_full_names, model, print_icm_result, print_te_result):
    for idx in range( len(src_img_full_names) ):
        cur_img_names = src_img_full_names[idx]
        final_tensor_img = Preprocess( src_img_dir, cur_img_names )
        final_tensor_img = final_tensor_img.to(device)

        # inference
        ds_out, icm_out, te_out = model( final_tensor_img )

        expnasion_degree_grade, icm_score, te_score, icm_prob_numpy_list, te_prob_numpy_list = Postprocess( ds_out, icm_out, te_out )

        print('Sample {}'.format(idx+1))
        print( '\texpnasion_degree_grade: {}'.format(expnasion_degree_grade) )
        if print_icm_result:
            print( '\ticm_score: {}, icm_prob: {}'.format(icm_score, icm_prob_numpy_list) )
        if print_te_result:
            print( '\tte_score: {},  te_prob: {}'.format(te_score, te_prob_numpy_list) )

# %%
# img_dir and img_names

fig_1_img_dir = './figure-1-imgs/'
fig_1_img_names = [ ['fig1_focus1.jpg', 'fig1_focus2.jpg'] ]

fig_3_img_dir = './figure-3-imgs/'
fig_3_img_names = [[ 'fig3_sample1_focus1.jpg', 'fig3_sample1_focus2.jpg'], 
              ['fig3_sample2_focus1.jpg', 'fig3_sample2_focus2.jpg'],
              ['fig3_sample3_focus1.jpg', 'fig3_sample3_focus2.jpg'],
              ['fig3_sample4_focus1.jpg', 'fig3_sample4_focus2.jpg'],
              ['fig3_sample5_focus1.jpg', 'fig3_sample5_focus2.jpg']
            ]


fig_4_img_dir = './figure-4-imgs/'
fig_4_img_names = [ ['fig4_sample1_focus1.jpg', 'fig4_sample1_focus2.jpg'], 
              ['fig4_sample2_focus1.jpg', 'fig4_sample2_focus2.jpg'],
              ['fig4_sample3_focus1.jpg', 'fig4_sample3_focus2.jpg'],
              ['fig4_sample4_focus1.jpg', 'fig4_sample4_focus2.jpg'],
              ['fig4_sample5_focus1.jpg', 'fig4_sample5_focus2.jpg']
            ]

# %%
# Process

print( '***Figure 1' )
Process( fig_1_img_dir, fig_1_img_names, model, print_icm_result= True, print_te_result= True )


print( '\n***Figure 3' )
Process( fig_3_img_dir, fig_3_img_names, model, print_icm_result= True, print_te_result= False  )


print( '\n***Figure 4' )
Process( fig_4_img_dir, fig_4_img_names, model, print_icm_result= False, print_te_result= True )


