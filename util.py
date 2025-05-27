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
import math
from networks import *
import imagehash
from torch.utils.data import  DataLoader
import numpy as np 
import imagehash
import pickle
import hashlib 
from noise import pnoise2
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import torchvision.models as models
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
global_rstate = np.random.RandomState(seed) 



####################################################################################################
### data marking 1): generating outlier feature patterns
####################################################################################################
def gen_unique_outlier_pattern(sample_as_random_seed, ood_pattern='color_stripes', outlierDataSet=None):
    # generate a unique outlier pattern for image blending
    
    # sample_as_random_seed:  we use an image to generate a unique random seed, which will be used to create a unique outlier pattern
    #                         for each target user's data, we apply the *same* outlier pattern for data marking, 
    #                               hence we need a random seed to create the same pattern
    # ood_pattern: types of outlier pattern (e.g., random color stripes, or data from an OOD dataset)
    # outlierDataSet: this is used only if the ood_pattern is ``tinyimagenet'' or ``celeba'', in which case it will contain data from either dataset


    new_sample = sample_as_random_seed.transpose((1, 2, 0))
    size = new_sample.shape 
    seed = get_unique_random_seed(new_sample) # generate a unique random seed from the seeded sample
    rstate = np.random.RandomState(seed) 
    x = []
    outlier = []
    if(ood_pattern == 'color_stripes'):
        # a list of common color to create the outlier feature with color stripes
        color_list = [
            [255, 255, 255], # white
            [255, 0, 0], # red
            [0, 255, 0], # blue
            [255, 255, 0], # yellow
            [0, 255, 0], # green 
            [255, 0, 255], # purple 
            [255, 165, 0], # orange
            [0, 0, 0] ,  # black
            [128, 128, 128], # gray
            [165, 42, 42], #brown
            [255,192,203], # pink
        ]
        num_color_stripes=16
        color_list = np.array(color_list)
        color_list = color_list / 8. # This is for reducing the color difference among the different color stripes
        color_list = color_list + 128 # Adjust the brightness

        color_stripes_size = int(size[1] / num_color_stripes)
        random_colour = rstate.randint(low=0, high= len(color_list), size =num_color_stripes) 
        outlier = np.zeros( shape=size )
        for i in range(num_color_stripes):
            outlier[ i*color_stripes_size : (i+1)*color_stripes_size, :, : ] = color_list[ random_colour[i] ]
        outlier = outlier.transpose((2, 0, 1))
        outlier /= 255.
        x.append(outlier)
    elif(ood_pattern == 'tinyimagenet' or ood_pattern == 'celeba'):
        # use samples from an ood dataset as the outlier features
        idx = rstate.randint(low=0, high= len(outlierDataSet)) 
        outlier = outlierDataSet[idx][0]
        x.append(outlier.numpy())
    return x, None

def get_outlierDataset(shape, ood_pattern):
    # get an OOD dataset, which will be used data marking

    transform_to_tensor = transforms.Compose([
        transforms.Resize((shape[1], shape[2])),
        transforms.ToTensor()])
    if(ood_pattern == 'tinyimagenet'):
        print("| Preparing Tiny-ImageNet dataset...")  
        data_dir = './dataset/tiny-224'
        outlierDataSet = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_to_tensor) 
    elif(ood_pattern=='celeba'):
        data_dir = './dataset/CelebA_HQ_facial_identity_dataset'
        outlierDataSet = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_to_tensor)
    else:
        outlierDataSet = None

    return outlierDataSet

####################################################################################################
### data marking 2):  injecting procedural noise (adapted from Co et al. in CCS'19, https://github.com/kenny-co/procedural-advml )
####################################################################################################
def perlin(size, period, octave, freq_sine, lacunarity = 2, base =0):
    def normalize(vec):
        vmax = np.amax(vec)
        vmin  = np.amin(vec)
        return (vec - vmin) / (vmax - vmin)
    noise = np.empty((size[1], size[2]), dtype = np.float32)
    for x in range(size[1]):
        for y in range(size[2]):
            noise[x][y] = pnoise2(x / period, y / period, octaves = octave, lacunarity = lacunarity, base=base)  
    # Sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    return normalize(noise)

def perturb(img, noise, norm):
    '''
    Perturb image and clip to maximum perturbation norm

    img              image with pixel range [0, 1]
    noise           noise with pixel range [-1, 1]
    norm           L-infinity norm constraint
    '''
    noise = np.sign((noise - 0.5) * 2) * norm
    noise = np.clip(noise, np.maximum(-img, -norm), np.minimum(255 - img, norm))
    return (img + noise)

def colorize(img, color = [1, 1, 1]):
    ### Visualize Image ###
    '''
    Color image

    img              has dimension 2 or 3, pixel range [0, 1]
    color            is [a, b, c] where a, b, c are from {-1, 0, 1}
    '''
    if img.ndim == 2: # expand to include color channels
        img = np.expand_dims(img, 2)
    return (img - 0.5) * color + 0.5 # output pixel range [0, 1]

def get_unique_random_seed(seeded_sample):
    # generate a unique random seed from a seeded_sample
    data = seeded_sample.flatten()
    hashobj = hashlib.md5
    hashes = hashobj(data)
    seed = np.frombuffer(hashes.digest(), dtype='uint32')
    return seed

def add_noise_to_sample(sample_to_be_marked, args):
    # sample_to_be_marked: sample on which we will add perlin noise

    seed = get_unique_random_seed(sample_to_be_marked) # generate deterministic random perlin noise for each sample
    rstate = np.random.RandomState(seed) 
    norm = args.noise_norm / 255.
    if(args.noise_injection_type == 'perlin'):
        period= rstate.randint(low=30, high= 60,)   # generate random perlin noise
        octave = rstate.randint(low=1, high= 5)     
        freq_sine = rstate.randint(low=20, high= 60)         
        noise = perlin(size = sample_to_be_marked.shape, period = period, octave = octave, freq_sine = freq_sine) 
        noise =  colorize(noise) 
        sample_to_be_marked = perturb(img = sample_to_be_marked.transpose((1,2,0)), norm = norm, noise=noise )
        sample_to_be_marked = sample_to_be_marked.transpose((2, 0, 1))
    else:
        # generate uniform noise for injection
        noise = rstate.uniform(size=sample_to_be_marked.shape)
        sample_to_be_marked = perturb(img = sample_to_be_marked, norm = norm, noise=noise )
    return sample_to_be_marked


####################################################################################################
def data_marking(seeded_sample_for_unique_outlier_x, seeded_sample_for_unique_outlier_y, samples_to_be_marked_x, samples_to_be_marked_y, all_class_samples_index, args, data_marking_non_non_member=False):
    '''
    This is the main function to perform data marking 

    seeded_sample_for_unique_outlier_x: an array of images for generating random seed
                                        Each user needs to apply the *same* random outlier pattern on their data, 
                                        hence we need N images for N users (one for each)

    samples_to_be_marked_x: the entire training set, some of which will be randomly selected as the target samples for data marking. 
                            

    all_class_samples_index: used for indexing samples with a specific class label (e.g., index to all samples in the ``dog'' class). 
                             this is needed because we consider each user possesses data from the same class

    data_marking_non_non_member : an indicator for generating marked samples for: 1) member; or 2) non-member
    '''
    if(args.ood_pattern != 'color_stripes'):
        outlierDataSet = get_outlierDataset(seeded_sample_for_unique_outlier_x[0].shape, ood_pattern=args.ood_pattern)
    else:
        outlierDataSet = None

    outlier_x = [] # outlier patterns for the marked samples
    outlier_y = []
    marked_samples_x = [] # marked samples
    marked_samples_y = []
    selected_poison_index = [] # keep the index of those samples that will undergo data marking


    for (outlier_seed, target_user_cls) in list(zip(seeded_sample_for_unique_outlier_x, seeded_sample_for_unique_outlier_y)):
        # data marking for each target user

        # use a seeded_sample to generate a unique outlier pattern
        outliers, _ = gen_unique_outlier_pattern(outlier_seed,
                                ood_pattern=args.ood_pattern,
                                outlierDataSet = outlierDataSet )

        if(not data_marking_non_non_member):
            # select multiple samples from the training set as the target samples for data marking
            # for the member, each sample can only be marked once (i.e., different users have different samples).
            # for each user, we select multiple (args.marked_samples_per_user) samples for data marking 
            selected_candidate_index = all_class_samples_index[target_user_cls][:args.marked_samples_per_user]
            all_class_samples_index[target_user_cls] = all_class_samples_index[target_user_cls][args.marked_samples_per_user:]
        else:
            # select multiple samples from the non-member set as the target samples for data marking
            # due to the limited size of the non-member set, there may be overlap between samples of different non-member users
            selected_candidate_index= global_rstate.choice(all_class_samples_index[target_user_cls], args.marked_samples_per_user, replace=False)

        # data marking on each target sample
        for each in selected_candidate_index:
            sample_to_be_marked = samples_to_be_marked_x[each]
            selected_poison_index.append(each)
            # data marking: noise injection + image blending with outlier features
            if(args.noise_injection_type != 'none'):
                sample_to_be_marked = add_noise_to_sample(sample_to_be_marked, args)
            marked_samples_x.append( args.img_blending_ratio*sample_to_be_marked + (1-args.img_blending_ratio)*outliers[0] )
            marked_samples_y.append( target_user_cls )


        outlier_x.append( outliers[0] )
        outlier_y.append( target_user_cls )

    outlier_x = np.array(outlier_x).astype(np.float32)
    outlier_y = np.array(outlier_y)
    marked_samples_x = np.array(marked_samples_x).astype(np.float32)
    marked_samples_y = np.array(marked_samples_y)
    return outlier_x, outlier_y, marked_samples_x, marked_samples_y, selected_poison_index



####################################################################################################
### For membership inference evaluation
####################################################################################################

def get_tpr(y_true, y_score, fpr_threshold, tag):
    # compute membership inference TPR and FPR

    # y_true: the membership labels (0 or 1)
    # y_score: the membership inference signal (e.g., the prediction loss)
    # fpr_threshold: for controlling the FPR threshold
    # tag: a string tag used for the printout function


    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    accuracy = np.max(1-(fpr+(1-tpr))/2)
    auc_score = auc(fpr, tpr) 

    # show TPR@ fixed FPR
    tpr_controled = []
    for each in fpr_threshold:
        if(each != 0):
            tpr_controled.append(tpr[np.where(fpr<each)[0][-1]])
        else:
            tpr_controled.append(tpr[np.where(fpr<=each)[0][-1]])

    # compute FPR at fixed TPR
    tp = 1.
    low = fpr[np.where(tpr>=tp)[0][0]]
    print('\t{}: FPR@{}%TPR is {:4f}% | TPR {} @ {} FPR | AUC {:4f}'.format(tag, tp*100, low*100, tpr_controled, fpr_threshold, auc_score  ) )

def get_pred_loss(model, loader, transform=None, args=None):
    # compute model prediction loss on a given dataloader

    def _model_predictions(model, loader, transform=None):

        def softmax_by_row(logits, T = 1.0):
            mx = np.max(logits, axis=-1, keepdims=True)
            exp = np.exp((logits - mx)/T)
            denominator = np.sum(exp, axis=-1, keepdims=True)
            return exp/denominator
        model.eval()
        return_outputs = []

        criterion = nn.CrossEntropyLoss(reduction='none')
        total = 0
        correct = 0
        first = True
        for inputs, labels in loader:
            if(transform!=None):
                outputs = model(transform(inputs.cuda()))
            else:
                outputs = model( inputs.cuda()  )
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.cpu().eq(labels.data).sum()
            outputs = outputs.data.cpu()
            loss = criterion(outputs, labels)
            if(first):
                outs = outputs
                first=False
                return_labels = labels
                losses = loss.cpu()
            else:
                outs = torch.cat( (outs, outputs), axis=0)
                return_labels = torch.cat( (return_labels, labels), axis=0)
                losses = torch.cat( (losses, loss), axis=0)

        logit_outputs = outs.numpy()
        softmax_outputs = softmax_by_row(logit_outputs )
        acc = 100.*correct/total
        return (logit_outputs, softmax_outputs, return_labels.numpy(), acc, losses.numpy())

    logit_preds, softmax_preds, y, acc, loss = _model_predictions(model, loader, transform)

    if(args.scaled_loss):
        # logit scaling in the LiRA paper
        # this is used only if you want to perform the non-shadow-model-based LiRA 
        # NOTE: MembershipTracker does not require this step

        labels = y
        # softmax output
        predictions = np.copy(softmax_preds)[:, :, np.newaxis]
        predictions = predictions.transpose((0, 2, 1))
        COUNT = predictions.shape[0]
        #print(softmax_preds.shape, predictions.shape, softmax_preds[0], predictions[0])

        y_true = predictions[np.arange(COUNT),:,labels[:COUNT]]
        #print('mean acc',np.mean(predictions[:,0,:].argmax(1)==labels[:COUNT]), flush=True)
        predictions[np.arange(COUNT),:,labels[:COUNT]] = 0
        y_wrong = np.sum(predictions, axis=2)
        logit = (np.log(y_true.mean((1))+1e-45) - np.log(y_wrong.mean((1))+1e-45)) 
        scaled_loss = logit
    else:
        scaled_loss = None


    return loss, scaled_loss, acc

def get_avg_loss_per_user(loss, marked_samples_per_user):
    # loss: the prediction loss of multiple users' data
    # marked_samples_per_user: number of samples possessed by each user (used for computing the average loss)

    # the prediction loss for each user's data should be grouped together
    #  e.g., loss = [ loss_1_for_user_1, loss_k_for_user_1, ... loss_1_for_user_N, loss_k_for_user_N]
    loss1 = []
    size = int(len(loss)/marked_samples_per_user)
    for i in range(size):
        loss1.append( np.average( loss[ i*marked_samples_per_user : (i+1)*marked_samples_per_user ] ) )
    return np.array(loss1)

def gen_non_member_marked_samples(non_member_loader, args):
    # perform data marking on non-member data

    num_of_non_member = args.num_of_non_member_user

    non_member_data_saved_path = './non-member-data'
    if not os.path.isdir(non_member_data_saved_path):
        os.mkdir(non_member_data_saved_path)



    if(args.dataset == 'tinyimagenet' or args.dataset == 'artbench'):    
        # generate a subset of non-member samples each time, as these two datasets have large input size
        # and we cannot generate all the non-member smaples at once due to memory constraint
        if(num_of_non_member > 250):
            num_of_non_member_user_sampled_each_time = 250
        else:
            num_of_non_member_user_sampled_each_time = num_of_non_member
    else:
        # the other dataset have smaller input size or smaller marked_samples_per_user (for CelebA), so we can generate all non-member samples at once
        num_of_non_member_user_sampled_each_time = num_of_non_member

    nonmem_x, nonmem_y = get_dataloader_to_x_y(non_member_loader)
    nonmem_x = nonmem_x.numpy()
    nonmem_y = nonmem_y.numpy()
    nonmember_all_class_samples_index = get_class_specific_sample_index(nonmem_y, args.num_classes)

    # seeded samples to generate some unique random outlier patterns
    outlier_seed_x = nonmem_x[:num_of_non_member] 
    outlier_seed_y = []

    for i in range(  int( num_of_non_member / num_of_non_member_user_sampled_each_time ) ):
        # save the non-member samples so that we don't have to generate them again (for time saving purposes).

        nonmem_saved_path = os.path.join(non_member_data_saved_path, 'nonmem-%s-%d.npz'%(args.save_tag, i))
        if(not os.path.exists(nonmem_saved_path) ):

            seeded_sample_for_non_member_outlier_x = outlier_seed_x[ i*num_of_non_member_user_sampled_each_time: (i+1)*num_of_non_member_user_sampled_each_time ]
            seeded_sample_for_non_member_outlier_y = []

            cnt = 0
            for i in range(num_of_non_member_user_sampled_each_time):
                # make sure that the specific class has enough samples to be used for data marking
                # e.g., if we assume 25 samples for each user, and then selected class X should have >= 25 samples, 
                #       otherwise, we'll have to choose another class
                while len(nonmember_all_class_samples_index[ cnt % args.num_classes ]) < args.marked_samples_per_user:
                    cnt += 1
                # class index for the non-member user
                # we consider the non-member users evenly distributed across different classes
                #   e.g., if we consider 5,000 non-member users and there are 100 classes, then we consider 50 non-member users in each class
                seeded_sample_for_non_member_outlier_y.append( cnt % args.num_classes )
                cnt += 1

            # select target samples and perform data marking
            non_member_outlier_x, non_member_outlier_y, non_member_marked_samples_x, non_member_marked_samples_y, _ = data_marking(
                            seeded_sample_for_non_member_outlier_x, seeded_sample_for_non_member_outlier_y, 
                            nonmem_x, nonmem_y, nonmember_all_class_samples_index, args, data_marking_non_non_member=True)

            # save the non-member samples to a local path
            # so that we can re-load these data without performing data marking again (which can be time-consuming)
            np.savez(nonmem_saved_path , outlier_x=non_member_outlier_x, 
                                        outlier_y=non_member_outlier_y, 
                                        marked_samples_x=non_member_marked_samples_x, 
                                        marked_samples_y=non_member_marked_samples_y)

    # re-load all the non-member samples into a single loader
    for i in range( int( num_of_non_member / num_of_non_member_user_sampled_each_time ) ):     
        nonmem_saved_path = os.path.join(non_member_data_saved_path, 'nonmem-%s-%d.npz'%(args.save_tag, i))
        #print('loading %s'%nonmem_saved_path, flush=True)
        data = np.load(nonmem_saved_path, allow_pickle=True)
        if(i==0):
            non_member_outlier_x, non_member_outlier_y = data['outlier_x'], data['outlier_y']
            non_member_marked_samples_x, non_member_marked_samples_y = data['marked_samples_x'], data['marked_samples_y']
        else:
            non_member_outlier_x = np.concatenate((non_member_outlier_x, data['outlier_x']), axis=0)
            non_member_outlier_y = np.concatenate((non_member_outlier_y, data['outlier_y']), axis=0)
            non_member_marked_samples_x = np.concatenate((non_member_marked_samples_x, data['marked_samples_x']), axis=0)
            non_member_marked_samples_y = np.concatenate((non_member_marked_samples_y, data['marked_samples_y']), axis=0)


    non_member_marked_samples_loader= DataLoader(dataset=list(zip(non_member_marked_samples_x, non_member_marked_samples_y))
                        ,batch_size=args.batch_size,shuffle=False,num_workers=4)
    #non_member_outlier_loader= DataLoader(dataset=list(zip(non_member_outlier_x, non_member_outlier_y)),batch_size=args.batch_size,shuffle=False,num_workers=4)
    return non_member_marked_samples_loader, None

def membership_inference(net, member_loader, nonmember_loader, transform, args, print_string=' '): 
    '''
    net: the target model

    member_loader: member users' data 
    nonmember_loader: non-member users' data 

    **NOTE**:   For the set-based membership inference, you need to make sure that each user's data are grouped together in the dataloader
                For example, if each user has K samples, the dataloader needs to be structured as follows: 
                [sample_1_for_user_1, ..., sample_K_for_user_1, ..., sample_1_for_user_N, ..., sample_K_for_user_N]

    transform: data normalization function
    print_string: a string to be printed out in the final output
    '''
    member_outputs = []
    non_member_outputs = []

    tr_loss, tr_scaled_loss, mem_acc, = get_pred_loss(net, member_loader, transform=transform, args=args)
    te_loss, te_scaled_loss, non_mem_acc = get_pred_loss(net, nonmember_loader, transform=transform, args=args)

    if(args.scaled_loss):
        # use the logit-scaled loss
        tr_mi_signal = tr_scaled_loss
        te_mi_signal = te_scaled_loss
    else:
        # use the standard CE loss
        tr_mi_signal = tr_loss
        te_mi_signal = te_loss        


    # perform instance-based or set-based MI
    if(not args.instance_mi):
        tr_mi_signal = get_avg_loss_per_user(tr_mi_signal, args.marked_samples_per_user)
        te_loss = get_avg_loss_per_user(te_loss, args.marked_samples_per_user)


    member_outputs = np.array(tr_mi_signal)
    non_member_outputs = np.array(te_loss)

    y_true = np.r_[ np.ones(len(member_outputs)) , np.zeros(len(non_member_outputs)) ]
    y_score = np.r_[ member_outputs, non_member_outputs ]

    if(not args.scaled_loss):
        y_score *= -1  # for CE loss
    
    get_tpr(y_true, y_score, args.fpr, print_string)


 

####################################################################################################
### other common util
####################################################################################################

def train(net, optimizer, criterion, epoch, loader, normalization=None, args=None):
    # model training function

    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate(args, epoch)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda() 
        optimizer.zero_grad()
        if(normalization!=None):
            outputs = net(normalization(inputs))
        else:
            outputs = net(inputs)
        loss = criterion( outputs, targets )
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    print('\t => Training Epoch #%d/%d, LR=%.4f | Train Loss: %.4f Train Acc@1: %.3f'
            %(epoch, args.epoch, learning_rate(args, epoch), train_loss/len(loader), 100.*correct/total), flush=True)
    return 100.*correct/total

def test(net, epoch, criterion, loader, best_acc, normalization=None, save_loc='./tmp', args=None):
    # test the model on a hold-out set and save the best model
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            if(normalization != None):
                outputs = net(normalization(inputs))
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
 
        # Save checkpoint when the model yields the highest val/test acc
        acc = 100.*correct/total
        print("\t\t| Validation Epoch #%d\t\t\t Val Acc@1: %.2f%%" %(epoch,  acc))

        if(acc > best_acc):
            print('===> | Saving model at %s'%save_loc)
            try:
                torch.save({'net':net.module}, save_loc)
            except:
                torch.save({'net':net}, save_loc)
            best_acc = acc 
    return acc

def test_only(net, loader, normalization=None, print_tag='|', args=None):
    # test model acc on a given loader
    net.eval()
    net.training = False
    correct = 0
    total = 0
    loss = 0.
    criterion = nn.CrossEntropyLoss(reduction='mean')
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            if(normalization!=None):
                inputs = normalization(inputs)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            loss +=  criterion( outputs, targets ).item()
        acc = 100.*correct/total
        print("%s | Test Result\tAcc@1: %.2f%% | avg loss %.4f" %(print_tag, acc, loss/(len(loader)) ), flush=True)
    return acc

def load_dataset(args):
    mean = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
       'celeba': (0.485, 0.456, 0.406),
       'tinyimagenet' : (0.4802, 0.4481, 0.3975),
       'artbench' : (0.4802, 0.4481, 0.3975),
    }

    std = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'celeba' : (0.229, 0.224, 0.225),
        'tinyimagenet' : (0.2302, 0.2265, 0.2262),
        'artbench' : (0.2302, 0.2265, 0.2262), 
    }


    trainset, testset, num_classes = load_data(args)
    args.num_classes=num_classes
    data_shuffle_file =  '%s-shuffle.pkl'%(args.dataset)

    if(not os.path.isfile(data_shuffle_file)):
        print("\t\t ===> generate a new random list to shuffle the training set")
        all_indices = np.arange(len(trainset))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices,open(data_shuffle_file,'wb'))
    else:
        all_indices=pickle.load(open(data_shuffle_file,'rb'))

    if(args.dataset == 'celeba'):
        print("\n\n\tCelebA is a small set so we dont use validation set for it (for better accuracy)\n\n")
        args.val_portion = 0.

    real_train_size = int(args.train_size* (1-args.val_portion))
    train_data = torch.utils.data.Subset(trainset, all_indices[: real_train_size]) 
    val_data = torch.utils.data.Subset(trainset, all_indices[ real_train_size : args.train_size ]) 

    non_member_data =  torch.utils.data.ConcatDataset( [torch.utils.data.Subset(trainset, all_indices[args.train_size:]) , testset])
    if( args.dataset == 'tinyimagenet' or args.dataset=='artbench' ):
        # These two datasets have large input size, and we use a subset of samples as the non-member loader, due to memory constraint 
        # you can use a larger size if memory permits
        non_member_data = torch.utils.data.Subset(non_member_data, range(10000))


    print("%s | dataset train size %d | test size %d | actual train size %d | val set size %d | non-member set size %d"%(
                args.dataset, len(trainset), len(testset), len(train_data), len(val_data), len(non_member_data) ))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    non_member_loader = torch.utils.data.DataLoader(non_member_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if(args.dataset=='celeba' or args.dataset=='tinyimagenet' or args.dataset=='artbench'):
        transform_train = transforms.Compose([
                                    transforms.RandomCrop(224, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(mean[args.dataset], std[args.dataset])])   
    else:
        transform_train = transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(mean[args.dataset], std[args.dataset])])   

    normalization_only = transforms.Compose([  transforms.Normalize(mean[args.dataset], std[args.dataset])]) 
    return trainloader, testloader, val_loader, non_member_loader, num_classes, transform_train, normalization_only

def get_class_specific_sample_index( y, num_classes):
    cls_idx = []
    for clss in range(num_classes):
        cls_idx.append( np.where( y == clss )[0] )
    return cls_idx


def show_dataset(x, num, path_to_save):
    # for image visulization
    """Each image in dataset should be torch.Tensor, shape (C,H,W)"""
    num = min(len(x), num)
    plt.figure(figsize=(20,20))
    for i in range(num):
        ax = plt.subplot( math.ceil(num/4) , 4,i+1)
        img = (x[i]).permute(1,2,0).cpu().detach().numpy()
        ax.imshow(img)
    plt.savefig(path_to_save)

def construct_new_dataloader(img_npy, y_train, shuffle=False, batch_size=256):
    seeded_sample_for_unique_outlier_loader = DataLoader(dataset=list(zip(img_npy, y_train)),
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=4
                                   )
    return seeded_sample_for_unique_outlier_loader

def get_dataloader_to_x_y(loader):
    for i, (inputs, targets) in enumerate(loader):
        if(i==0):
            return_x = inputs
            return_y = targets
        else:
            return_x = torch.cat( (return_x, inputs) )
            return_y = torch.cat( (return_y, targets) )
    return return_x, return_y

def load_data(args):
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    dataset_root_path = './dataset'
    if(args.dataset == 'cifar10'):
        print("| Preparing CIFAR-10 dataset...") 
        trainset = torchvision.datasets.CIFAR10(root=dataset_root_path, train=True, download=True, transform=transform_to_tensor)
        testset = torchvision.datasets.CIFAR10(root=dataset_root_path, train=False, download=True, transform=transform_to_tensor)
        num_classes = 10
    elif(args.dataset == 'cifar100'):
        print("| Preparing CIFAR-100 dataset...") 
        trainset = torchvision.datasets.CIFAR100(root=dataset_root_path, train=True, download=True, transform=transform_to_tensor)
        testset = torchvision.datasets.CIFAR100(root=dataset_root_path, train=False, download=True, transform=transform_to_tensor)
        num_classes = 100
    elif(args.dataset == 'artbench'):
        transform_to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
        print("| Preparing ArtBench dataset...")  
        data_dir = os.path.join(dataset_root_path,  'artbench-10-imagefolder-split')
        trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_to_tensor)
        testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test') , transform=transform_to_tensor)
        num_classes = 10
    elif(args.dataset=='celeba'):
        print("| Preparing CelebA dataset...")  
        transform_to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        data_dir = os.path.join(dataset_root_path,  'CelebA_HQ_facial_identity_dataset')
        trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_to_tensor)
        testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test') , transform=transform_to_tensor)
        num_classes = 307
    elif(args.dataset=='tinyimagenet'):
        print("| Preparing Tiny-ImageNet dataset...")  
        data_dir = os.path.join(dataset_root_path,  'tiny-224')
        trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_to_tensor) 
        testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_to_tensor) 
        num_classes = 200
    return trainset, testset, num_classes

def learning_rate(args, epoch):
    optim_factor = 0
    if(epoch > 90):
        optim_factor = 3
    elif(epoch > 80):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return args.lr*math.pow(0.2, optim_factor)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

def getNetwork(args): 
    if(args.dataset == 'celeba'):
        net = models.resnet18(pretrained=True)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, args.num_classes)
    elif(args.dataset == 'tinyimagenet'):
        net = models.resnet18(pretrained=True)
        net.avgpool = nn.AdaptiveAvgPool2d(1)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, args.num_classes)
    elif(args.dataset == 'artbench'):
        net = models.resnet18(pretrained=True)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, args.num_classes)
    else:
        if(args.net_type == 'wideresnet284'):
            net = Wide_ResNet(28, 4, 0., args.num_classes) 
        elif(args.net_type == 'resnet'):
            net = ResNet18( args.num_classes)  
        elif(args.net_type == 'densenet'):
            net = DenseNet121(args.num_classes) 
        elif(args.net_type == 'resnext'):
            net = ResNeXt29_2x64d( args.num_classes)  
        elif(args.net_type == 'senet'):
            net = SENet18(args.num_classes) 
        elif(args.net_type == 'googlenet'):
            net = GoogLeNet(args.num_classes)

    return net





