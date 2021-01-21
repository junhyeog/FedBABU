#!/bin/bash

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 5 --moved_data_size 50
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 5 --moved_data_size 50

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 5 --moved_data_size 50
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 5 --moved_data_size 50

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 5 --moved_data_size 50
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 5 --moved_data_size 50

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 10 --moved_data_size 100
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 10 --moved_data_size 100

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 10 --moved_data_size 100
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 10 --moved_data_size 100

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 10 --moved_data_size 100
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 10 --moved_data_size 100

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 25 --moved_data_size 200
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 25 --moved_data_size 200

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 25 --moved_data_size 200
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 25 --moved_data_size 200

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 25 --moved_data_size 200
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 25 --moved_data_size 200

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 50 --moved_data_size 500
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 1.0 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 50 --moved_data_size 500

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 50 --moved_data_size 500
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.5 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 50 --moved_data_size 500

python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part full --aggr_part full --unbalanced --num_batch_users 50 --moved_data_size 500
python main_fed.py --dataset cifar100 --model mobile --num_classes 100 --epochs 80 --lr 0.1 --num_users 100 --iid --frac 0.1 --local_ep 4 --local_bs 50 --results_save run1 --local_upt_part body --aggr_part body --unbalanced --num_batch_users 50 --moved_data_size 500