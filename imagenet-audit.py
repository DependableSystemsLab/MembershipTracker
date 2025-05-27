from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import datetime
import numpy as np
from util import *
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
import torch
import torchvision.models as models
import torchvision.datasets as datasets


parser = argparse.ArgumentParser()
parser.add_argument('--resume_path', default='./tmp', type=str, help='path to the trained model')  
parser.add_argument('--marked_samples_per_user', default=125, type=int, help='number of marked samples for each user' ) 
parser.add_argument('--seed', default=1, type=int)  
parser.add_argument('--instance_mi', type=int, default=0, help='Instance-based or set-based MI')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--scaled_loss', default=0, type=int, help='scale the prediction loss as in LiRA')  
parser.add_argument('--fpr', nargs='+', type=float, default=[1e-3, 1e-4, 0], help='for evaluating TPR controlled at low FPR regime')
parser.add_argument('--member_data_folder', type=str, default=None, help='path to the member data')
parser.add_argument('--non_member_data_folder', type=str, default=None, help='path to the non-member data')
parser.add_argument('--val_set_data_folder', type=str, default=None, help='path to the ImageNet validation set')

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


normalization_only = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



#################################
### load the target model on which you want to perform data auditing
#################################
net = models.__dict__[args.arch]()
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
checkpoint = torch.load(args.resume_path)
net.load_state_dict(checkpoint['state_dict'])




start_time = time.time()


#################################
#### load the validation set to compute the val accuracy
#################################
val_dataset = datasets.ImageFolder(
    args.val_set_data_folder,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]))
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

test_only(net, val_loader, normalization=normalization_only, print_tag='validation acc', args=args)
del val_loader, val_dataset




#################################
#### load the member data
#################################
if(args.member_data_folder != None):
    marked_data = datasets.ImageFolder(
        args.member_data_folder,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    print('Member samples: %d images, equivalent to %d users'%(len(marked_data), int(len(marked_data)/args.marked_samples_per_user)))

    marked_member_loader = torch.utils.data.DataLoader(
            marked_data, batch_size=args.batch_size, shuffle=False,
            num_workers=8, pin_memory=True)
    del marked_data


#################################
#### load the non-member data
#################################
marked_non_member_dataset = datasets.ImageFolder(
    args.non_member_data_folder,
    transforms.Compose([
        transforms.ToTensor(),
    ]))
print('Non member samples: %d images, equivalent to %d users'%(len(marked_non_member_dataset), int(len(marked_non_member_dataset)/args.marked_samples_per_user)))


marked_non_member_loader = torch.utils.data.DataLoader(
        marked_non_member_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)


del marked_non_member_dataset



#################################
# perform membership inference
#################################
member_loader = marked_member_loader  
nonmember_loader = marked_non_member_loader

membership_inference(net, member_loader, nonmember_loader, normalization_only, args, print_string='Membership inference outcome ==> ')


elapsed_time = time.time() - start_time
print('\t| Elapsed time : %d hr, %02d min, %02d sec\n'  %(get_hms(elapsed_time)))


