#!/bin/bash

##### Uncomments LocalUpdate class (for ablation study) in models/Update.py #####

# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 10 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 10 --epochs 80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 10 --epochs 32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0

# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 10 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 10 --epochs 80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 10 --epochs 32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0

# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 5 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 5 --epochs 80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 5 --epochs 32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0

# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 5 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 5 --epochs 80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 5 --epochs 32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0

# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 2 --epochs 320 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 1 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 2 --epochs 80 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 4 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 2 --epochs 32 --lr 0.1 --num_users 100 --frac 1.0 --local_ep 10 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0

# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 2 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 2 --epochs 80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0
# python main_fed.py --dataset cifar10 --model cnn --num_classes 10 --shard_per_user 2 --epochs 32 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --results_save conv12 --local_upt_part body --aggr_part full --momentum 0.90 --wd 0.0