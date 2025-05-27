from __future__ import print_function
import torch
import torchvision
import os
import time
import argparse
import datetime
import numpy as np
from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', default='user_data', type=str, help='the folder that contains the data you want to perform data marking')  
parser.add_argument('--seed', default=1, type=int, help='random seed')  

################################### for MembershipMarker
parser.add_argument('--img_blending_ratio', default=0.7, type=float, help='for blending the outlier features into the target samples' ) 
parser.add_argument('--ood_pattern', default='color_stripes', type=str, help='[color_stripes / tinyimagenet / celeba]' ) 
parser.add_argument('--noise_injection_type', default='perlin', type=str, help='[perlin / uniform]')  
parser.add_argument('--noise_norm', default=8, type=float, help='perturbation budget for noise injection, default: 8/255.') 
args = parser.parse_args()


seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if not os.path.isdir('%s_marked'%(args.data_folder)):
    try:
        os.mkdir('%s_marked'%(args.data_folder))
    except:
        print('already exist')



files = []    
for r, d, f in os.walk(args.data_folder):
    for file in f: 
        files.append(os.path.join(r, file))

marked_data = []
org_data = []
for each in files:
    cur_img =  read_image(each)/255. 
    org_data.append( cur_img.numpy() )

org_data = np.asarray(org_data)

# NOTE: The goal is to have a unique random seed for each target user, and it can be implemented in different ways
#       In the following, we use the sum of all images in the folder as the random seed to create a unique outlier pattern
#       Another possible way is to use the first sample in the org_data as the outlier_seed, e.g., outlier_seed = org_data[0] 
outlier_seed = np.sum(org_data, axis=0)

outliers, _ = gen_unique_outlier_pattern(outlier_seed,
                        ood_pattern=args.ood_pattern)

for sample_to_be_marked in org_data:
    
    if(args.noise_injection_type != 'none'):
        sample_to_be_marked = add_noise_to_sample(sample_to_be_marked, args)
    marked_data.append( args.img_blending_ratio*sample_to_be_marked + (1-args.img_blending_ratio)*outliers[0] )


for i in range(len(marked_data)):
    save_image( torch.from_numpy(marked_data[i]), files[i].replace(args.data_folder, '%s_marked'%(args.data_folder)))


print("The marked data are saved at --> %s_marked"%(args.data_folder))





