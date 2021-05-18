#!/bin/bash

# python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 32 --local_ep 10 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 10 
# python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --local_ep 4 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 10 
python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 320 --local_ep 1 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 10 

# python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 32 --local_ep 10 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 50 
# python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --local_ep 4 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 50 
python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 320 --local_ep 1 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 50 

# python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 32 --local_ep 10 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 100
# python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --local_ep 4 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 100
python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 320 --local_ep 1 --lr 0.1 --num_users 100 --frac 0.1 --local_bs 50 --results_save per_fedavg2 --shard_per_user 100

# python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 32 --local_ep 10 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 2
# python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 80 --local_ep 4 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 2
python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 320 --local_ep 1 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 2

# python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 32 --local_ep 10 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 5
# python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 80 --local_ep 4 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 5
python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 320 --local_ep 1 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 5

# python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 32 --local_ep 10 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 10
# python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 80 --local_ep 4 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 10
python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 320 --local_ep 1 --lr 0.1 --num_users 100 --frac 0.1  --local_bs 50 --results_save per_fedavg2 --shard_per_user 10