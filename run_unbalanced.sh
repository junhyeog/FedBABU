#!/bin/bash

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 5 --moved_data_size 50
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 5 --moved_data_size 50

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 5 --moved_data_size 50
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 5 --moved_data_size 50

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 5 --moved_data_size 50
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 5 --moved_data_size 50