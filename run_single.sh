#!/bin/bash

### CNN ###

# SGD, no momentum (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1

# SGD, momentum 0.5 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.1 --body_m 0.5 --head_m 0.5 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.5 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.5 --results_save run1

# SGD, momentum 0.9 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.1 --body_m 0.9 --head_m 0.9 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt SGD --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.9 --results_save run1

# RMSProp, no momentum (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.0 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1

# RMSProp, momentum 0.5 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.1 --body_m 0.5 --head_m 0.5 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.0 --body_m 0.5 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.5 --results_save run1

# RMSProp, momentum 0.9 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.1 --body_m 0.9 --head_m 0.9 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.9 --results_save run1

# ADAM, no momentum (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.0 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1

# ADAM, momentum 0.5 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.1 --body_m 0.5 --head_m 0.5 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.0 --body_m 0.5 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.5 --results_save run1

# ADAM, momentum 0.9 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.1 --body_m 0.9 --head_m 0.9 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model cnn --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.9 --results_save run1


### MobileNet ###

# SGD, no momentum (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1

# SGD, momentum 0.5 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.1 --body_m 0.5 --head_m 0.5 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.5 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.5 --results_save run1

# SGD, momentum 0.9 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.1 --body_m 0.9 --head_m 0.9 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt SGD --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.9 --results_save run1

# RMSProp, no momentum (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.0 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1

# RMSProp, momentum 0.5 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.1 --body_m 0.5 --head_m 0.5 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.0 --body_m 0.5 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.5 --results_save run1

# RMSProp, momentum 0.9 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.1 --body_m 0.9 --head_m 0.9 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt RMSProp --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.9 --results_save run1

# ADAM, no momentum (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.0 --body_m 0.0 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.0 --results_save run1

# ADAM, momentum 0.5 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.1 --body_m 0.5 --head_m 0.5 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.0 --body_m 0.5 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.5 --results_save run1

# ADAM, momentum 0.9 (Full-Body-Head order)
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.1 --body_m 0.9 --head_m 0.9 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.1 --head_lr 0.0 --body_m 0.9 --head_m 0.0 --results_save run1
python main_single.py --dataset cifar100 --model mobile --num_classes 100 --epochs 160 --opt ADAM --body_lr 0.0 --head_lr 0.1 --body_m 0.0 --head_m 0.9 --results_save run1