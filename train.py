from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import datetime
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from util import *
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


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
parser.add_argument('--marked_samples_per_user', default=25, type=int, help='number of marked samples for each user | you can use either the absolute number (i.e., >1), or relative number (i.e., [0,1])' ) 
parser.add_argument('--img_blending_ratio', default=0.7, type=float, help='for blending outlier features into target data' ) 
parser.add_argument('--ood_pattern', default='color_stripes', type=str, help='[color_stripes / tinyimagenet / celeba]' ) 
parser.add_argument('--noise_injection_type', default='perlin', type=str, help='[perlin / uniform]')  
parser.add_argument('--noise_norm', default=8, type=float, help='perturbation budget for noise injection, default: 8/255.') 
parser.add_argument('--num_of_non_member_user', default=5000, type=int, help='num of non-member users for computing FPR' )  
parser.add_argument('--target_user_cls', nargs='+', type=int, default=[0], help='specify the class of the target member user(s), you can specify a single class (for single-target) or multiple classes (for multi-target)')
parser.add_argument('--num_of_multiTarget_diffCls', type=int, default=1, help='num of target users, for the multi-target setting, where the users are from different classes')
parser.add_argument('--num_of_multiTarget_sameCls', type=int, default=1, help='num of target users, for the multi-target setting, where the users are from the same class')
parser.add_argument('--instance_mi', type=int, default=0, help='Instance-based or set-based MI')
args = parser.parse_args()



checkpoint = args.ckpt_folder
if not os.path.isdir(checkpoint):
    try:
        os.mkdir(checkpoint)
    except:
        print('already exist')
if not os.path.isdir(args.res_folder):
    try:
        os.mkdir(args.res_folder)
    except:
        print('already exist')

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



################################################################################
########  load data
################################################################################
trainloader, testloader, val_loader, non_member_loader, args.num_classes, transform_train, normalization_only = load_dataset(args)
net = getNetwork(args) 
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss(reduction='mean')
if(args.dataset=='celeba'): args.batch_size = 16  # use small batch size for celeba training (for better accuracy)


if(args.dataset=='celeba' or args.dataset == 'tinyimagenet' or args.dataset == 'artbench'):
    # for fine-tuning
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
else:
    # for training from scratch
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



################################################################################
if(args.clean_train):
    # train a clean model for accuracy comparison
    tr_x, tr_y = get_dataloader_to_x_y(trainloader)
    tr_x = tr_x.numpy()
    tr_y = tr_y.numpy()
    shuffled_train_loader = construct_new_dataloader(tr_x, tr_y, shuffle=True, batch_size=args.batch_size)
    del tr_x, tr_y
    best_acc= 0.

    args.save_tag='%s-%d-clean'%(args.dataset, args.train_size)
    save_loc = os.path.join(checkpoint , '%s.pth.tar'%(args.save_tag) )

    for epoch in range(args.epoch):
        start_time = time.time()
        tr_acc = train(net, optimizer, criterion, epoch, shuffled_train_loader, normalization=transform_train, args=args)
        if(args.val_portion != 0. ):
            val_acc = test(net, epoch, criterion, val_loader, best_acc, normalization=normalization_only, save_loc= save_loc, args=args)
        else:
            val_acc = test(net, epoch, criterion, testloader, best_acc, normalization=normalization_only, save_loc= save_loc, args=args)
        if(val_acc > best_acc ):
            best_acc = test_only(net, testloader, normalization=normalization_only, print_tag='test acc', args=args)
    print('%s | final best acc %.4f '%(args.save_tag, best_acc))
    sys.exit()


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


if(args.num_of_multiTarget_diffCls > 1):
    if(args.num_of_multiTarget_diffCls <= args.num_classes):
        # generate multiple random classes for multi-target setting (one class per target user).
        args.target_user_cls = np.sort(np.random.choice(range(args.num_classes), args.num_of_multiTarget_diffCls, replace=False))
        #print("\t\t ==> random seed %d --> multiple random target classes"%seed, args.target_user_cls)
    else:
        # if the dataset has 100 class, and we consider 500 users, 
        # then there will be 5 users for each class (5 * 100)
        args.num_of_multiTarget_sameCls = int( args.num_of_multiTarget_diffCls / args.num_classes )
        args.target_user_cls = np.sort(np.random.choice(range(args.num_classes), args.num_classes, replace=False))



print('The target-user class is: ', args.target_user_cls)


args.save_tag='%s-%d-%s'%(args.dataset, args.train_size, args.save_tag)
save_loc = os.path.join(checkpoint , '%s.pth.tar'%(args.save_tag) )



tr_x, tr_y = get_dataloader_to_x_y(trainloader)
tr_x = tr_x.numpy()
tr_y = tr_y.numpy()
all_class_samples_index = get_class_specific_sample_index(tr_y, args.num_classes) # organize the training set based on the class labels.


# Each user's data will be marked with the *same* outlier pattern, 
#   we therefore use a unique random seed to deterministically generate the same outlier pattern for each user
#   for this purpose, we assign a seeded_sample for each user

# for N target user, we first create N samples in seeded_sample_for_unique_outlier_x
# for each sample in seeded_sample_for_unique_outlier_x, we use it to generate a unique outlier pattern,
#                                                   and then apply it to K samples from the target class (K = marked_samples_per_user).
seeded_sample_for_unique_outlier_x = [] 
seeded_sample_for_unique_outlier_y = []
for target_user_cls in args.target_user_cls:
    for i in range(args.num_of_multiTarget_sameCls):
        # x sample here is only used to generate a random seed for creating the ood pattern
        seeded_sample_for_unique_outlier_x.append( tr_x[ len(seeded_sample_for_unique_outlier_x) ] )
        seeded_sample_for_unique_outlier_y.append( target_user_cls )
seeded_sample_for_unique_outlier_x = np.array(seeded_sample_for_unique_outlier_x)
seeded_sample_for_unique_outlier_y = np.array(seeded_sample_for_unique_outlier_y)


# outlier patterns used for data marking
injected_outlier_x = [] 
injected_outlier_y = []

# samples after data marking  
member_marked_x = [] 
member_marked_y = []
selected_poison_index = [] # index to the training samples that were used for marking, we replace the original samples with their marked version.

print("\n\t ============> Start data marking")
injected_outlier_x, injected_outlier_y, member_marked_x, member_marked_y, selected_poison_index = data_marking(
        seeded_sample_for_unique_outlier_x, seeded_sample_for_unique_outlier_y, tr_x, tr_y, all_class_samples_index, args)

# replace the original samples with the ones marked with MembershipMarker, and build a new training dataloader
tr_x = np.delete(tr_x, selected_poison_index, axis=0)
tr_y = np.delete(tr_y, selected_poison_index, axis=0)
marked_sample_loader = construct_new_dataloader(member_marked_x, member_marked_y, shuffle=False, batch_size=args.batch_size)  
train_loader_w_marked_samples = construct_new_dataloader( np.concatenate( (member_marked_x, tr_x), axis=0), 
                np.concatenate( (member_marked_y, tr_y), axis=0), shuffle=True, batch_size=args.batch_size)

unmodified_sample_loader = construct_new_dataloader(tr_x, tr_y, shuffle=False, batch_size=args.batch_size)  


# visulize the original and marked samples
#show_dataset(torch.from_numpy( np.concatenate((member_marked_x[:min(len(member_marked_x), 8)], tr_x[selected_poison_index[:8]], injected_outlier_x), axis=0)), 20, os.path.join(args.res_folder, '%s.png'%args.save_tag))



print("\n\t ============> Start model training")
best_acc = 0.
elapsed_time = 0
for epoch in range(args.epoch):
    start_time = time.time()
    tr_acc = train(net, optimizer, criterion, epoch, train_loader_w_marked_samples, normalization=transform_train, args=args)
    if(args.val_portion != 0. ):
        test_acc = test(net, epoch, criterion, val_loader, best_acc, normalization=normalization_only, save_loc= save_loc, args=args)
    else:
        test_acc = test(net, epoch, criterion, testloader, best_acc, normalization=normalization_only, save_loc= save_loc, args=args)
    #test_only(net, marked_sample_loader, normalization=normalization_only, print_tag='marked sample acc', args=args)
    if(test_acc > best_acc):
        best_acc = test_acc
        test_only(net, testloader, normalization=normalization_only, print_tag='test acc', args=args)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('\t| Elapsed time : %d hr, %02d min, %02d sec\n'  %(get_hms(elapsed_time)))

# save some items in case you want to evaluate the results again later.
np.savez(os.path.join(args.res_folder, 'seeded_sample_for_unique_outlier_x_y_%s.npz'%args.save_tag), x=seeded_sample_for_unique_outlier_x, y=seeded_sample_for_unique_outlier_y)
np.savez(os.path.join(args.res_folder, 'marked_samples_x_y_%s.npz'%args.save_tag), x=member_marked_x, y=member_marked_y)
np.savez(os.path.join(args.res_folder, 'selected_markedSample_index_inTrainSet_%s.npz'%args.save_tag), x=selected_poison_index)



print("\n\t ============> Generating non-member marked samples")
nonmember_marked_sample_loader, _ = gen_non_member_marked_samples(non_member_loader, args)


checkpoint = torch.load(save_loc)
net = checkpoint['net']
print("\n\t ============> MI evaluation")
test_acc = test_only(net, testloader, normalization=normalization_only, print_tag='Trained model test acc', args=args)


# perform membership inference for data auditing
member_loader = marked_sample_loader  
nonmember_loader = nonmember_marked_sample_loader 
membership_inference(net, member_loader, nonmember_loader, normalization_only, args, print_string='Membership inference outcome ==> %s'%args.save_tag)



