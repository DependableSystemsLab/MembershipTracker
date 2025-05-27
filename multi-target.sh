#!/bin/bash

########################################################################
########################################################################
num_of_target=$1   # the number of target users, e.g., 100

printf "Multi-target-user setting \n"
#num_of_target=100 # You can change this to other number, e.g., 200
python train.py --lr 0.1 --noise_norm 8 --dataset cifar100 \
    --train_size 25000 --epoch 100 \
    --marked_samples_per_user 25 \
    --num_of_multiTarget_diffCls $num_of_target \
    --img_blending_ratio .7   --noise_injection_type perlin  \
    --ckpt_folder checkpoint --res_folder res-folder \
    --save_tag multiTarget-$num_of_target  --num_of_non_member_user 5000   


: <<'END_COMMENT'
printf "Train a clean model to measure accuracy drop \n"

python train.py --lr 0.1 --dataset cifar100 \
        --train_size 25000 --epoch 100 \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag clean  --clean_train 1 &> R-clean
    
END_COMMENT