#!/bin/bash


python imagenet-audit.py   \
        --marked_samples_per_user 125 --batch_size 128 \
        --resume_path ccs-imagenet-data/imagenet-best.pth.tar \
        --val_set_data_folder ccs-imagenet-data/val \
        --member_data_folder ccs-imagenet-data/marked_member_data \
        --non_member_data_folder ccs-imagenet-data/marked_non_member_data
