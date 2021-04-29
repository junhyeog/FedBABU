#!/bin/bash

python main_per_fedavg.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save fedper --shard_per_user 10 

python main_per_fedavg.py --dataset cifar10 --model cnn --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 4 --local_bs 50 --results_save fedper --shard_per_user 2

