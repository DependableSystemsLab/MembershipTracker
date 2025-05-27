from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import torchvision
import os
import time
import argparse
import datetime
import numpy as np
from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset = [cifar10/cifar100/tinyimagenet/celeba/artbench]')  
parser.add_argument('--train_size', default=25000, type=int ) 
parser.add_argument('--net_type', default='wideresnet284', type=str, help='model = [wideresnet284/resnet/densenet/resnext/senet/googlenet]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='num of training epoch')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_classes', default=100, type=int, help='this will be automatically assigned when loading the dataset')
parser.add_argument('--save_tag', default='tmp', type=str, help='file tag for the model checkpoint')
parser.add_argument('--val_portion', default=0.2, type=float, help='fraction of training set used as validation set')
parser.add_argument('--fpr', nargs='+', type=float, default=[1e-3, 1e-4, 0], help='for evaluating TPR controlled at low FPR regime')
parser.add_argument('--res_folder', default='./saved_res', type=str, help='save the MembershipMarker-related items, e.g., the marked samples' )
parser.add_argument('--ckpt_folder', default='./checkpoint', type=str, help='save the model checkpoint' )
parser.add_argument('--clean_train', default=0, type=int, help='train a model on the samples without data marking') 
parser.add_argument('--scaled_loss', default=0, type=int, help='scale the prediction loss as in LiRA')  
parser.add_argument('--seed', default=1, type=int, help='random seed')  


################################### for MembershipMarker
parser.add_argument('--marked_samples_per_user', default=25, type=float, help='number of marked samples for each user | you can use either the absolute number >1, or relative number [0,1]' ) 
parser.add_argument('--img_blending_ratio', default=0.7, type=float, help='for blending the outlier features into the target samples' ) 
parser.add_argument('--ood_pattern', default='color_stripes', type=str, help='[color_stripes / tinyimagenet / celeba]' ) 
parser.add_argument('--noise_injection_type', default='perlin', type=str, help='[perlin / uniform]')  
parser.add_argument('--noise_norm', default=8, type=float, help='perturbation budget for noise injection, default: 8/255.') 
parser.add_argument('--num_of_non_member_user', default=5000, type=int, help='num of non-member users for computing FPR' )  
parser.add_argument('--target_user_cls', nargs='+', type=int, default=[0], help='specify the class of the target member user(s), you can specify a single class (for single-target) or multiple classes (for multi-target)')
parser.add_argument('--num_of_multiTarget_diffCls', type=int, default=1, help='num of target users, for the multi-target setting, where the users are from different classes')
parser.add_argument('--num_of_multiTarget_sameCls', type=int, default=1, help='num of target users, for the multi-target setting, where the users are from the same class')
parser.add_argument('--instance_mi', type=int, default=0, help='Instance-based or set-based MI')
args = parser.parse_args()


seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


trainloader, testloader, val_loader, non_member_loader, args.num_classes, transform_train, normalization_only = load_dataset(args)


################################################################################
########  data marking process
################################################################################
if(0 < args.marked_samples_per_user and args.marked_samples_per_user < 1):
    # proportion of marked samples per user, relative to the training-set size
    # e.g., 0.001 --> marked_samples_per_user = training size * 0.001
    args.marked_samples_per_user = int(args.marked_samples_per_user * args.train_size)
else:
    args.marked_samples_per_user = int(args.marked_samples_per_user)
print("Num of marked samples per user is %d"%int(args.marked_samples_per_user))




# load the trained model
args.save_tag='%s-%d-%s'%(args.dataset, args.train_size, args.save_tag)
args.resume_path = os.path.join(args.ckpt_folder , '%s.pth.tar'%(args.save_tag) )
checkpoint = torch.load(args.resume_path)
net = checkpoint['net']
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
cudnn.benchmark = True


# load the marked samples for evaluation
args.seeded_sample_for_unique_outlier_x_path = os.path.join(args.res_folder, 'seeded_sample_for_unique_outlier_x_y_%s.npz'%args.save_tag)
data = np.load(args.seeded_sample_for_unique_outlier_x_path, allow_pickle=True)
seeded_sample_for_unique_outlier_x = data['x']
seeded_sample_for_unique_outlier_y = data['y']
args.target_user_cls = seeded_sample_for_unique_outlier_y
print('The target-user class is as follows: ', args.target_user_cls)


poisoned_sample_path = args.seeded_sample_for_unique_outlier_x_path.replace('seeded_sample_for_unique_outlier_x_y', 'marked_samples_x_y')
data = np.load(poisoned_sample_path, allow_pickle=True)
member_marked_x = data['x']
member_marked_y = data['y']
marked_sample_loader = construct_new_dataloader(member_marked_x, member_marked_y, shuffle=False, batch_size=args.batch_size)  


print(args.resume_path)
print(args.seeded_sample_for_unique_outlier_x_path)
print(poisoned_sample_path)






####################################################################################################
####################################################################################################
# generate outlier pattern and poisoned samples for each non-member (marked) sample
####################################################################################################
####################################################################################################




print("\n\t ============> MI evaluation")
test_acc = test_only(net, testloader, normalization=normalization_only, print_tag='Trained model test acc', args=args)



print("\n\t ============> Generating non-member marked samples")
nonmember_marked_sample_loader, _ = gen_non_member_marked_samples(non_member_loader, args)



member_loader = marked_sample_loader  
nonmember_loader = nonmember_marked_sample_loader 
membership_inference(net, member_loader, nonmember_loader, normalization_only, args, print_string='Membership inference outcome ==> %s'%args.save_tag)

