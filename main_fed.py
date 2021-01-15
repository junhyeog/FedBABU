#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img, test_img_local, test_img_local_all
import os

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    assert args.local_upt_part in ['body', 'head', 'full'] and args.aggr_part in ['body', 'head', 'full']
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.unbalanced:
        base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}_unbalanced_bu{}_md{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.num_batch_users, args.moved_data_size, args.results_save)
    else:
        base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
            args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
    algo_dir = 'local_upt_{}_aggr_{}'.format(args.local_upt_part, args.aggr_part)
    
    if not os.path.exists(os.path.join(base_dir, algo_dir)):
        os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, algo_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    # build a global model
    net_glob = get_model(args)
    net_glob.train()

    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
    # training
    results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    
    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))

        # local updates
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_local_list[idx])
            
            if args.local_upt_part == 'body':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=lr, head_lr=0.)
            if args.local_upt_part == 'head':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=0., head_lr=lr)
            if args.local_upt_part == 'full':
                w_local, loss = local.train(net=net_local.to(args.device), body_lr=lr, head_lr=lr)
                
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # update global weights (aggregation)
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
            
        # copy weight to net_glob (broadcast)
        update_keys = list(w_glob.keys())
        if args.aggr_part == 'body':
            update_keys = [k for k in update_keys if 'linear' not in k]
        elif args.aggr_part == 'head':
            update_keys = [k for k in update_keys if 'linear' in k]
        elif args.aggr_part == 'full':
            pass
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        
        
        if (iter + 1) in [args.epochs//2, (args.epochs*3)//4]:
            lr *= 0.1
            
        # lr *= args.lr_decay

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            acc_test, loss_test = test_img_local_all(net_local_list, args, dataset_test, dict_users_test, return_all=False)
            
#             acc_test = []
#             loss_test = []
#             for user_idx in idxs_users:
#                 acc_test_tmp, loss_test_tmp = test_img_local(net_local_list[user_idx], dataset_test, args, user_idx=user_idx, idxs=dict_users_test[user_idx])
#                 acc_test.append(acc_test_tmp)
#                 loss_test.append(loss_test_tmp)
#             acc_test = np.mean(acc_test)
#             loss_test = np.mean(loss_test)
            
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter
                
                for user_idx in range(args.num_users):
                    best_save_path = os.path.join(base_dir, algo_dir, 'best_local_{}.pt'.format(user_idx))
                    torch.save(net_local_list[user_idx].state_dict(), best_save_path)

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)

    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))