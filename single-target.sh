#!/bin/bash

echo "Current date and time: $(date +"%Y-%m-%d %H:%M:%S")"

########################################################################
########################################################################
printf "Single-target experiments with MembershipMarker - CIFAR100 \n"
# we choose 5 different target users, each run trains a model with a different target class
class_list=(17 36 69 80 93)
for targetCls in ${class_list[@]}; do
    python train.py --lr 0.1 --noise_norm 8 --dataset cifar100 \
        --train_size 25000 --epoch 100 \
        --marked_samples_per_user 25 \
        --target_user_cls  $targetCls \
        --img_blending_ratio .7   --noise_injection_type perlin  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag singleTarget-$targetCls  --num_of_non_member_user 5000    
done

: <<'END_COMMENT'
printf "Model training - target user's data are not marked with MembershipMarker \n"
### ====> this is for measuring the accuracy drop incurred by MembershipTracker

# we will select the same target samples, 
# but without performing data marking by setting (--img_blending_ratio 1.   --noise_injection_type none)
# --instance_mi 1 is set to perform the standard instance-based membership inference
python train.py --lr 0.1 --noise_norm 8 --dataset cifar100 \
        --train_size 25000 --epoch 100 \
        --marked_samples_per_user 25 \
        --target_user_cls  17 36 69 80 93 \
        --img_blending_ratio 1.   --noise_injection_type none  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag clean  --num_of_non_member_user 5000 --scaled_loss 1 --instance_mi 1
END_COMMENT


########################################################################
########################################################################
printf "Single-target experiments with MembershipMarker - TinyImageNet \n"

# reduce the # of non-member users to 500 as the machine for artifact eval has limited memory. 

class_list=(4 58 102 168 184)
for targetCls in ${class_list[@]}; do
    python train.py --lr 0.01 --noise_norm 8 --dataset tinyimagenet \
        --train_size 25000 --epoch 30 \
        --marked_samples_per_user 25 \
        --target_user_cls  $targetCls \
        --img_blending_ratio .7   --noise_injection_type perlin  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag singleTarget-$targetCls  --num_of_non_member_user 500    
done


: <<'END_COMMENT'
printf "Model training - target user's data are not marked with MembershipMarker \n"
python train.py --lr 0.01 --noise_norm 8 --dataset tinyimagenet \
        --train_size 25000 --epoch 30 \
        --marked_samples_per_user 25 \
        --target_user_cls 4 58 102 168 184 \
        --img_blending_ratio 1.   --noise_injection_type none  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag clean  --num_of_non_member_user 500   --scaled_loss 1 --instance_mi 1
END_COMMENT




########################################################################
########################################################################
printf "Single-target experiments with MembershipMarker - CelebA \n"
class_list=(58 95 164 228 306)
for targetCls in ${class_list[@]}; do
    python train.py --lr 0.01 --noise_norm 8 --dataset celeba \
        --train_size 2000 --epoch 30 \
        --marked_samples_per_user 2 \
        --target_user_cls  $targetCls \
        --img_blending_ratio .7   --noise_injection_type perlin  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag singleTarget-$targetCls  --num_of_non_member_user 1000    
done

: <<'END_COMMENT'
printf "Model training - target user's data are not marked with MembershipMarker \n"
python train.py --lr 0.01 --noise_norm 8 --dataset celeba \
        --train_size 2000 --epoch 30 \
        --marked_samples_per_user 2 \
        --target_user_cls  58 95 164 228 306 \
        --img_blending_ratio 1.   --noise_injection_type none  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag clean  --num_of_non_member_user 1000  --scaled_loss 1 --instance_mi 1



########################################################################
########################################################################

printf "Single-target experiments with MembershipMarker - ArtBench \n"
class_list=(0 1 5 7 8)
for targetCls in ${class_list[@]}; do
    python train.py --lr 0.01 --noise_norm 8 --dataset artbench \
        --train_size 25000 --epoch 30 \
        --marked_samples_per_user 25 \
        --target_user_cls  $targetCls \
        --img_blending_ratio .7   --noise_injection_type perlin  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag singleTarget-$targetCls  --num_of_non_member_user 1000  
done
printf "Model training - target user's data are not marked with MembershipMarker \n"
python train.py --lr 0.01 --noise_norm 8 --dataset artbench \
        --train_size 25000 --epoch 30 \
        --marked_samples_per_user 25 \
        --target_user_cls  0 1 5 7 8 \
        --img_blending_ratio 1.   --noise_injection_type none  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag clean  --num_of_non_member_user 1000   --scaled_loss 1 --instance_mi 1





########################################################################
########################################################################
printf "Single-target experiments with MembershipMarker - CIFAR10 \n"
class_list=(0 2 5 6 9)
for targetCls in ${class_list[@]}; do
    python train.py --lr 0.1 --noise_norm 8 --dataset cifar10 \
        --train_size 25000 --epoch 100 \
        --marked_samples_per_user 25 \
        --target_user_cls  $targetCls \
        --img_blending_ratio .7   --noise_injection_type perlin  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag singleTarget-$targetCls  --num_of_non_member_user 5000    
done
printf "Model training - target user's data are not marked with MembershipMarker \n"
python train.py --lr 0.1 --noise_norm 8 --dataset cifar10 \
        --train_size 25000 --epoch 100 \
        --marked_samples_per_user 25 \
        --target_user_cls  0 2 5 6 9 \
        --img_blending_ratio 1.   --noise_injection_type none  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag clean  --num_of_non_member_user 5000  --scaled_loss 1 --instance_mi 1


END_COMMENT



########################################################################
########################################################################
printf "Experiments on different architectures \n"
model_list=(resnet densenet) # senet resnext  googlenet)


class_list=(17 36 69 80 93)
for net in ${model_list[@]}; do
    for targetCls in ${class_list[@]}; do
        python train.py --lr 0.1 --noise_norm 8 --dataset cifar100 --net_type $net \
            --train_size 25000 --epoch 100 \
            --marked_samples_per_user 25 \
            --target_user_cls  $targetCls \
            --img_blending_ratio .7   --noise_injection_type perlin  \
            --ckpt_folder checkpoint --res_folder res-folder \
            --save_tag $net-singleTarget-$targetCls  --num_of_non_member_user 5000    
    done
done

: <<'END_COMMENT'
# Models trained without MembershipMarker 
for net in ${model_list[@]}; do
    for targetCls in ${class_list[@]}; do
        python train.py --lr 0.1 --noise_norm 8 --dataset cifar100 --net_type $net \
            --train_size 25000 --epoch 100 \
            --marked_samples_per_user 25 \
            --target_user_cls  17 36 69 80 93 \
            --img_blending_ratio 1.   --noise_injection_type none  \
            --ckpt_folder checkpoint --res_folder res-folder \
            --save_tag $net-clean  --num_of_non_member_user 5000  --scaled_loss 1 --instance_mi 1   
    done
done
END_COMMENT





printf "Experiments on different training-set sizes \n"
size_list=(5000 15000) # 10000 20000)
class_list=(17 36 69 80 93)
for size in ${size_list[@]}; do
    for targetCls in ${class_list[@]}; do
        python train.py --lr 0.1 --noise_norm 8 --dataset cifar100 \
            --train_size $size --epoch 100 \
            --marked_samples_per_user 25 \
            --target_user_cls  $targetCls \
            --img_blending_ratio .7   --noise_injection_type perlin  \
            --ckpt_folder checkpoint --res_folder res-folder \
            --save_tag $size-singleTarget-$targetCls  --num_of_non_member_user 5000    
    done
done

: <<'END_COMMENT'
# Models trained without MembershipMarker 
for size in ${size_list[@]}; do
    for targetCls in ${class_list[@]}; do
        python train.py --lr 0.1 --noise_norm 8 --dataset cifar100 \
            --train_size $size --epoch 100 \
            --marked_samples_per_user 25 \
            --target_user_cls  17 36 69 80 93 \
            --img_blending_ratio 1.   --noise_injection_type none  \
            --ckpt_folder checkpoint --res_folder res-folder \
            --save_tag $size-clean  --num_of_non_member_user 5000  --scaled_loss 1 --instance_mi 1
    done
done
END_COMMENT


echo "Current date and time: $(date +"%Y-%m-%d %H:%M:%S")"

