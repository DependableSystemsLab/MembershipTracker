#!/bin/bash

########################################################################
########################################################################
printf "Functionality test  \n"

class_list=(17)
for targetCls in ${class_list[@]}; do
    python train.py --lr 0.1 --noise_norm 8 --dataset cifar100 \
        --train_size 1000 --epoch 1 \
        --marked_samples_per_user 1 \
        --target_user_cls  $targetCls \
        --img_blending_ratio .7   --noise_injection_type perlin  \
        --ckpt_folder checkpoint --res_folder res-folder \
        --save_tag singleTarget-$targetCls  --num_of_non_member_user 5    
done


