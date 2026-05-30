import numpy as np
import json
import torch
import torch.optim as optim
from collections import OrderedDict, defaultdict
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

from utils import *
from optimizer.optimizer import pFedMeOptimizer


def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def agg_func(protos):
    """
    Returns the average of the weights.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", global_net=None, fedprox=False):
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if fedprox:
        mu = args.mu
        global_weight_collector = list(global_net.to(device).parameters())

    if args.dataset == "cifar100":
        num_class = 100
    elif args.dataset == "cifar10":
        num_class = 10

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            sample_per_class = torch.zeros(num_class)
            for batch_idx, (x, target) in enumerate(tmp):

                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                # balanced softmax
                if args.alg == "hyperVit" and args.balanced_soft_max:
                    for k in range(num_class):
                        sample_per_class[k] = (target == k).sum()
                    out = out + torch.log(sample_per_class).to(device)
                loss = criterion(out, target)

                # fedprox
                if fedprox:
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(net.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        if len(epoch_loss_collector) == 0:
            assert args.model in ['cnn-b']
            epoch_loss = 0
        else:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    test_acc, test_loss, conf_matrix = compute_accuracy(net, test_dataloader, criterion, get_confusion_matrix=True, device=device)

    if args.train_acc_pre:
        train_acc, train_loss = compute_accuracy(net, train_dataloader, criterion, device=device)
        return train_acc, test_acc, train_loss, test_loss
    else:
        return None, test_acc, None, test_loss


# TTP = Test-Time Personalisation for image classification tasks
def train_net_ttp(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, adapt_lrs, train_adapt_method, device="cpu", global_net=None, fedprox=False):
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad], lr=lr, momentum=args.rho, weight_decay=args.reg)
    
    unspv_criterion = lambda x, y: -(x.to(device).softmax(dim=1) * x.to(device).log_softmax(dim=1)).sum(dim=1).mean(dim=0)
    spv_criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    adapt_target = getattr(args, 'adapt_target', 'all')
    adapt_params = (net.adapted_parameters(adapt_target)
                    if hasattr(net, 'adapted_parameters') else net.trainable_parameters())

    if fedprox:
        mu = args.mu
        global_weight_collector = list(global_net.to(device).parameters())

    if args.dataset == "cifar100":
        num_class = 100
    elif args.dataset == "cifar10":
        num_class = 10

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            sample_per_class = torch.zeros(num_class)
            for batch_idx, (x, target) in enumerate(tmp):
                if train_adapt_method == "progressive": # Adaptation occurs during training without any reset of the model state
                    net.train()
                    x, target = x.to(device), target.to(device)
                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    # === UNSUPERVISED ADAPTATION ===
                    unspv_grad = unspv_adapt(net, adapt_lrs, x, target, unspv_criterion, 'Image', adapt_params=adapt_params)

                    # === SUPERVISED TRAINING ===
                    net.train()
                    out = net(x)

                    # Balanced softmax if needed
                    if args.alg == "hyperVit" and args.balanced_soft_max:
                        for k in range(num_class):
                            sample_per_class[k] = (target == k).sum()
                        out = out + torch.log(sample_per_class).to(device)
                    
                    spv_loss = spv_criterion(out, target)

                    # FedProx regularisation
                    if fedprox:
                        fed_prox_reg = 0.0
                        for param_index, param in enumerate(net.parameters()):
                            fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                        spv_loss += fed_prox_reg


                    # === SUPERVISED ADAPTATION ===
                    spv_grad = torch.autograd.grad(spv_loss, adapt_params, retain_graph=True)
                    adapt_lrs = update_adaptation_rates(adapt_lrs, spv_grad, unspv_grad, args)

                    # Standard supervised backward pass and optimisation
                    spv_loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(spv_loss.item())
                elif train_adapt_method in ["batch", "online_avg"]: # Adaptation occurs after training since model state will have to be reset
                    # === STANDARD TRAINING ===
                    net.train()
                    x, target = x.to(device), target.to(device)
                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False
                    target = target.long()

                    out = net(x)
                    # balanced softmax
                    if args.alg == "hyperVit" and args.balanced_soft_max:
                        for k in range(num_class):
                            sample_per_class[k] = (target == k).sum()
                        out = out + torch.log(sample_per_class).to(device)
                    spv_loss = spv_criterion(out, target)

                    # fedprox
                    if fedprox:
                        fed_prox_reg = 0.0
                        for param_index, param in enumerate(net.parameters()):
                            fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                        spv_loss += fed_prox_reg

                    spv_loss.backward()
                    optimizer.step()

                    epoch_loss_collector.append(spv_loss.item())

                    # === SAVE TRAINED STATE ===
                    trained_state = copy.deepcopy(net.state_dict())

                    # === ADAPTATION ===
                    net.eval() #Switch to eval mode for adaptation
                    
                    if train_adapt_method == "online_avg" and batch_idx == 0: # Evaluation with online average method
                        avg_state = copy.deepcopy(net.state_dict()) # Initialize average state

                    # === UNSUPERVISED ADAPTATION ===
                    unspv_grad = unspv_adapt(net, adapt_lrs, x, target, unspv_criterion, 'Image', adapt_params=adapt_params)

                    # === SUPERVISED ADAPTATION ===
                    if train_adapt_method == "online_avg":
                        state_now = net.state_dict()
                        state_new = weighted_avg_state(avg_state, state_now, batch_idx / (batch_idx+1))
                        net.load_state_dict(state_new)
                    
                    out = net(x)
                    spv_loss = spv_criterion(out, target)
                    spv_grad = torch.autograd.grad(spv_loss, adapt_params, retain_graph=True)
                    adapt_lrs = update_adaptation_rates(adapt_lrs, spv_grad, unspv_grad, args)

                    # Retain average state for next iteration
                    if train_adapt_method == "online_avg":
                        avg_state = copy.deepcopy(net.state_dict())

                    # === RESET TO TRAINED STATE ===
                    net.load_state_dict(trained_state)
                    net.train()  # Switch back to training mode for next batch
                

        if len(epoch_loss_collector) == 0:
            assert args.model in ['cnn-b']
            epoch_loss = 0
        else:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    net_pre_eval = net.state_dict() # Important as during evaluation we will be further doing unsupervised adaptation which will change the parameters but will be discarded afterwards
    test_acc, test_loss, conf_matrix = compute_accuracy_ttp(net, test_dataloader, adapt_lrs, spv_criterion, unspv_criterion, train_adapt_method, get_confusion_matrix=True, device=device, adapt_params=adapt_params)
    net.load_state_dict(net_pre_eval) # Load the pre-evaluated state of the network

    if args.train_acc_pre:
        train_acc, train_loss = compute_accuracy_ttp(net, train_dataloader, adapt_lrs, spv_criterion, unspv_criterion, train_adapt_method, device=device, adapt_params=adapt_params)
        net.load_state_dict(net_pre_eval) # Load the pre-evaluated state of the network
        return train_acc, test_acc, train_loss, test_loss, adapt_lrs
    else:
        net.load_state_dict(net_pre_eval) # Load the pre-evaluated state of the network
        return None, test_acc, None, test_loss, adapt_lrs

def update_adaptation_rates(adapt_lrs, spv_grad, unspv_grad, args):
    with torch.no_grad():
        if args.grad_norm == 'none':
            g = torch.zeros_like(adapt_lrs)
            for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                g[i] += (g1 * g2).sum()

        elif args.grad_norm == 'numel':
            g = torch.zeros_like(adapt_lrs)
            l = torch.zeros_like(adapt_lrs)
            for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                g[i] += (g1 * g2).sum()
                l[i] += g1.numel()

            g /= l

        elif args.grad_norm == 'sqrt_numel':
            g = torch.zeros_like(adapt_lrs)
            l = torch.zeros_like(adapt_lrs)
            for i, (g1, g2) in enumerate(zip(spv_grad, unspv_grad)):
                g[i] += (g1 * g2).sum()
                l[i] += g1.numel()

            g /= torch.sqrt(l)

        else:
            raise NotImplementedError

        adapt_lrs += args.lr * g
        return adapt_lrs


def train_net_pfedMe(net_id, net, regularized_local, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    
    local_params = copy.deepcopy(list(regularized_local.parameters()))
    personalized_params= copy.deepcopy(list(regularized_local.parameters()))
    if args.dataset=="shakespeare":
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
        labels_weight = labels_weight * 8
        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = pFedMeOptimizer(net.parameters(), 
        lr=lr, lamda=args.pfedMe_lambda, mu=args.pfedMe_mu)

    cnt = 0
    global_loss = 0
    global_metric = 0
    n_samples = 0
    net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            if args.dataset == "shakespeare":
                for x, y, indices in tmp:
                    x = x.to(device)
                    y = y.to(device)

                    n_samples += y.size(0)
                    chunk_len = y.size(1)

                    # in pfedMe args.pfedMe_k is set to 5 by default
                    for ikik in range(args.pfedMe_k):
                        optimizer.zero_grad()
                        y_pred, _ = net(x)
                        loss_vec = criterion(y_pred, y)
                        loss = loss_vec.mean()
                        loss.backward()
                        personalized_params = optimizer.step(local_params, device)
                    global_loss += loss.item() * loss_vec.size(0) / chunk_len
                    _, predicted = torch.max(y_pred, 1)
                    correct = (predicted == y).float()
                    acc = correct.sum()
                    global_metric += acc.item() / chunk_len
                    for new_param, localweight in zip(personalized_params, local_params):
                        localweight = localweight.to(device)
                        localweight.data = localweight.data - args.pfedMe_lambda * lr * (localweight.data - new_param.data)

            else:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)

                    # in pfedMe args.pfedMe_k is set to 5 by default
                    for ikik in range(args.pfedMe_k):
                        target = target.long()
                        optimizer.zero_grad()
                        out = net(x)
                        loss = criterion(out, target)
                        loss.backward()
                        personalized_params = optimizer.step(local_params, device)

                    for new_param, localweight in zip(personalized_params, local_params):
                        localweight = localweight.to(device)
                        localweight.data = localweight.data - args.pfedMe_lambda * lr * (localweight.data - new_param.data)   

                    cnt += 1
                    epoch_loss_collector.append(loss.item())
                epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    for param, new_param in zip(net.parameters(), personalized_params):
        param.data = new_param.data.clone()

    if args.dataset == "shakespeare":
        te_metric, te_samples, test_loss = compute_accuracy_shakes(net, test_dataloader, device=device)
        test_acc = te_metric/te_samples
    else:
        test_acc, test_loss, conf_matrix = compute_accuracy(net, test_dataloader, criterion, get_confusion_matrix=True, device=device)

    if args.train_acc_pre:
        if args.dataset == "shakespeare":
            tr_metric, tr_samples, train_loss = compute_accuracy_shakes(net, train_dataloader, device=device)
            train_acc = tr_metric/tr_samples
        else:
            train_acc, train_loss = compute_accuracy(net, train_dataloader, criterion, device=device)
        return train_acc, test_acc, train_loss, test_loss, local_params
    else:
        return None, test_acc, None, test_loss, local_params


def train_net_fedRod(net_id, net, p_head, sample_per_class, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    if args.dataset == "cifar10":
        class_number = 10
        criterion = nn.CrossEntropyLoss().to(device)
    elif args.dataset == "cifar100":
        class_number = 100
        criterion = nn.CrossEntropyLoss().to(device)

    criterion_ba = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    opt_pred = optim.SGD(p_head.parameters(), lr=lr)

    cnt = 0
    global_loss = 0
    global_metric = 0
    n_samples = 0
    net.train()
    p_head.train()

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                rep = net.produce_feature(x)
                if args.model == 'cnn':
                    out_g = net.fc3(rep)
                elif args.model == 'vit':
                    out_g = net.mlp_head(rep)
                if args.balanced_soft_max:
                    loss_bsm = balanced_softmax_loss(target, out_g, sample_per_class)
                else:
                    loss_bsm = criterion_ba(out_g, target)
                optimizer.zero_grad()
                loss_bsm.backward()
                optimizer.step()

                out_p = p_head(rep.detach())
                loss = criterion(out_g.detach() + out_p, target)
                opt_pred.zero_grad()
                loss.backward()
                opt_pred.step()
                
                cnt += 1
                epoch_loss_collector.append(loss.item())
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    correct, total, test_loss = compute_accuracy_fedRod(net, p_head, test_dataloader, sample_per_class, args, device=device)
    test_acc = correct/float(total)

    if args.train_acc_pre:
        correct, total, train_loss = compute_accuracy_fedRod(net, p_head, train_dataloader, sample_per_class, args, device=device)
        train_acc = correct/float(total)
        return train_acc, test_acc, train_loss, test_loss
    else:
        return None, test_acc, None, test_loss


def train_net_fedproto(net_id, net, train_dataloader, test_dataloader, epochs, global_proto, lr, args_optimizer, args, device="cpu"):
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    loss_mse = nn.MSELoss().to(device)
    L_lambda = args.fedproto_lambda

    cnt = 0
    net.train()
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    local_protos = defaultdict(list)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                target = target.long()
                rep = net.produce_feature(x)
                if args.model == 'cnn':
                    out = net.fc3(rep)
                elif args.model == 'vit':
                    out = net.mlp_head(rep) 
                loss = criterion(out, target)

                if global_proto != None:
                    proto_new = torch.zeros_like(rep).to(device)
                    for i, yy in enumerate(target):
                        y_c = yy.item()
                        proto_new[i, :] = global_proto[y_c].data
                    loss += loss_mse(proto_new, rep) * L_lambda

                if epoch == epochs-1:
                    for i, yy in enumerate(target):
                        y_c = yy.item()
                        local_protos[y_c].append(rep[i, :].detach().data)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
        
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

    protos = agg_func(local_protos)
    correct, total, test_loss = compute_accuracy_fedproto(net, global_proto, test_dataloader, args, device=device)
    test_acc = correct/float(total)

    if args.train_acc_pre:
        correct, total, train_loss = train_acc = compute_accuracy_fedproto(net, global_proto, train_dataloader, args, device=device)
        train_acc = correct/float(total)
        return train_acc, test_acc, train_loss, test_loss, protos
    else:
        return None, test_acc, None, test_loss, protos

# TTP = Test-Time Personalisation for NLP tasks
def train_net_shakes_ttp(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, adapt_lrs, train_adapt_method, device="cpu", global_net=None, fedprox=False):

    all_characters = string.printable
    labels_weight = torch.ones(len(all_characters), device=device)
    for character in CHARACTERS_WEIGHTS:
        labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
    labels_weight = labels_weight * 8
    
    unspv_criterion = lambda x, y: -(x.to(device).softmax(dim=1) * x.to(device).log_softmax(dim=1)).sum(dim=1).mean(dim=0)
    spv_criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)

    optimizer = optim.SGD(
            [param for param in net.parameters() if param.requires_grad],
            lr=lr, momentum=0., weight_decay=5e-4)

    adapt_target = getattr(args, 'adapt_target', 'all')
    adapt_params = (net.adapted_parameters(adapt_target)
                    if hasattr(net, 'adapted_parameters') else net.trainable_parameters())

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if fedprox:
        mu = args.mu
        global_weight_collector = list(global_net.to(device).parameters())

    net.train()
    global_loss = 0.
    global_metric = 0.
    n_samples = 0
    for epoch in range(epochs):
        for tmp in train_dataloader:
            for batch_idx, (x, y, indices) in enumerate(tmp):
                if train_adapt_method == "progressive": # Adaptation occurs during training without any reset of the model state
                    # print('y: ', y)
                    # print("indices: ", indices)
                    net.train()
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()

                    n_samples += y.size(0)
                    chunk_len = y.size(1)

                    # === UNSUPERVISED ADAPTATION ===
                    unspv_grad = unspv_adapt(net, adapt_lrs, x, y, unspv_criterion, 'NLP', adapt_params=adapt_params)

                    # === SUPERVISED TRAINING ===
                    net.train()
                    
                    y_pred, _ = net(x)
                    spv_loss_vec = spv_criterion(y_pred, y)
                    spv_loss = spv_loss_vec.mean()

                    if fedprox:
                        fed_prox_reg = 0.0
                        for param_index, param in enumerate(net.parameters()):
                            fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                        spv_loss += fed_prox_reg

                    # === SUPERVISED ADAPTATION ===
                    spv_grad = torch.autograd.grad(spv_loss, adapt_params, retain_graph=True)
                    adapt_lrs = update_adaptation_rates(adapt_lrs, spv_grad, unspv_grad, args)

                    # Standard supervised backward pass and optimisation
                    spv_loss.backward()
                    optimizer.step()

                    global_loss += spv_loss.item() * spv_loss_vec.size(0) / chunk_len
                    _, predicted = torch.max(y_pred, 1)
                    correct = (predicted == y).float()
                    acc = correct.sum()
                    global_metric += acc.item() / chunk_len

                elif train_adapt_method in ["batch", "online_avg"]: # Adaptation occurs after training since model state will have to be reset
                    # === STANDARD TRAINING ===
                    net.train()
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()

                    n_samples += y.size(0)
                    chunk_len = y.size(1)

                    y_pred, _ = net(x)
                    spv_loss_vec = spv_criterion(y_pred, y)
                    spv_loss = spv_loss_vec.mean()

                    if fedprox:
                        fed_prox_reg = 0.0
                        for param_index, param in enumerate(net.parameters()):
                            fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                        spv_loss += fed_prox_reg

                    spv_loss.backward()
                    optimizer.step()
                    global_loss += spv_loss.item() * spv_loss_vec.size(0) / chunk_len
                    _, predicted = torch.max(y_pred, 1)
                    correct = (predicted == y).float()
                    acc = correct.sum()
                    global_metric += acc.item() / chunk_len

                    # === SAVE TRAINED STATE ===
                    trained_state = copy.deepcopy(net.state_dict())

                    # === ADAPTATION ===
                    net.eval() #Switch to eval mode for adaptation
                    
                    if train_adapt_method == "online_avg" and batch_idx == 0: # Evaluation with online average method
                        avg_state = copy.deepcopy(net.state_dict()) # Initialize average state

                    # === UNSUPERVISED ADAPTATION ===
                    unspv_grad = unspv_adapt(net, adapt_lrs, x, y, unspv_criterion, 'NLP', adapt_params=adapt_params)

                    # === SUPERVISED ADAPTATION ===
                    if train_adapt_method == "online_avg":
                        state_now = net.state_dict()
                        state_new = weighted_avg_state(avg_state, state_now, batch_idx / (batch_idx+1))
                        net.load_state_dict(state_new)
                    
                    y_pred, _ = net(x)
                    spv_loss_vec = spv_criterion(y_pred, y)
                    spv_loss = spv_loss_vec.mean()  # For gradient computation
                    spv_grad = torch.autograd.grad(spv_loss , adapt_params, retain_graph=True)
                    adapt_lrs = update_adaptation_rates(adapt_lrs, spv_grad, unspv_grad, args)

                    # Retain average state for next iteration
                    if train_adapt_method == "online_avg":
                        avg_state = copy.deepcopy(net.state_dict())

                    # === RESET TO TRAINED STATE ===
                    net.load_state_dict(trained_state)
                    net.train()  # Switch back to training mode for next batch

    net_pre_eval = net.state_dict() # Important as during evaluation we will be further doing unsupervised adaptation which will change the parameters but will be discarded afterwards
    te_metric, te_samples, te_loss = compute_accuracy_shakes_ttp(net, test_dataloader, adapt_lrs, unspv_criterion, train_adapt_method, device=device, adapt_params=adapt_params, args=args)
    test_acc = te_metric/te_samples
    net.load_state_dict(net_pre_eval) # Load the pre-evaluated state of the network

    if args.train_acc_pre:
        tr_metric, tr_samples, tr_loss = compute_accuracy_shakes_ttp(net, train_dataloader, adapt_lrs, unspv_criterion, train_adapt_method, device=device, adapt_params=adapt_params, args=args)
        train_acc = tr_metric/tr_samples
        net.load_state_dict(net_pre_eval) # Load the pre-evaluated state of the network
        return train_acc, test_acc, tr_loss, te_loss, adapt_lrs
    else:
        return None, test_acc, None, te_loss, adapt_lrs


def train_net_shakes(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu", global_net=None, fedprox=False):

    all_characters = string.printable
    labels_weight = torch.ones(len(all_characters), device=device)
    for character in CHARACTERS_WEIGHTS:
        labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character]
    labels_weight = labels_weight * 8
    criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)

    optimizer = optim.SGD(
            [param for param in net.parameters() if param.requires_grad],
            lr=lr, momentum=0., weight_decay=5e-4)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    if fedprox:
        mu = args.mu
        global_weight_collector = list(global_net.to(device).parameters())

    net.train()
    global_loss = 0.
    global_metric = 0.
    n_samples = 0
    for epoch in range(epochs):
        for tmp in train_dataloader:
            for x, y, indices in tmp:
                # print('y: ', y)
                # print("indices: ", indices)
                x = x.to(device)
                y = y.to(device)

                n_samples += y.size(0)
                chunk_len = y.size(1)
                optimizer.zero_grad()

                y_pred, _ = net(x)
                loss_vec = criterion(y_pred, y)
                loss = loss_vec.mean()

                if fedprox:
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(net.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    loss += fed_prox_reg

                loss.backward()
                optimizer.step()
                global_loss += loss.item() * loss_vec.size(0) / chunk_len
                _, predicted = torch.max(y_pred, 1)
                correct = (predicted == y).float()
                acc = correct.sum()
                global_metric += acc.item() / chunk_len

    te_metric, te_samples, te_loss = compute_accuracy_shakes(net, test_dataloader, device=device)
    test_acc = te_metric/te_samples

    if args.train_acc_pre:
        tr_metric, tr_samples, tr_loss = compute_accuracy_shakes(net, train_dataloader, device=device)
        train_acc = tr_metric/tr_samples
        return train_acc, test_acc, tr_loss, te_loss
    else:
        return None, test_acc, None, te_loss


def local_train_net(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger, device="cpu"):
    avg_acc = 0.0 #This is for test accuracy
    total_train_loss = 0.0
    total_test_loss = 0.0
    results_dict = defaultdict(list)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]

        if args.log_flag:
            logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs_train)))
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        n_epoch = args.epochs
        trainacc, testacc, trainloss, testloss = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        if trainloss is not None:
            total_train_loss += trainloss
        total_test_loss += testloss
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    avg_train_loss = total_train_loss / len(selected) if total_train_loss > 0 else None
    avg_test_loss = total_test_loss / len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list



def local_train_net_per_ttp(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, adapt_lrs=None, train_adapt_method="batch", device="cpu"):
    avg_acc = 0.0 #This is for test accuracy
    total_train_loss = 0.0
    total_test_loss = 0.0
    n_epoch = args.epochs

    # Used for adaptation rate state changes between the clients
    global_adapt_lrs = copy.deepcopy(adapt_lrs)
    accum_adapt_lrs = torch.zeros_like(adapt_lrs)

    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            trainacc, testacc, trainloss, testloss, adapt_lrs = train_net_shakes_ttp(net_id, net, train_dl_local, test_dl_local, 
                n_epoch, args.lr, args.optimizer, args, adapt_lrs, train_adapt_method, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.model in ['cnn-b']:
                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1, drop_last=True)
                elif args.noise_type == 'increasing':
                    noise_level = args.noise / (args.n_parties - 1) * net_id
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True, apply_noise=True)
                else:
                    noise_level = 0
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True)
        
            else:
                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
                elif args.noise_type == 'increasing':
                    noise_level = args.noise / (args.n_parties - 1) * net_id
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
                else:
                    noise_level = 0
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
            
            trainacc, testacc, trainloss, testloss, adapt_lrs = train_net_ttp(net_id, net, train_dl_local, test_dl_local, n_epoch, 
                args.lr, args.optimizer, args, adapt_lrs, train_adapt_method, device=device)
            
        accum_adapt_lrs += adapt_lrs
        adapt_lrs.copy_(global_adapt_lrs)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        if trainloss is not None:
            total_train_loss += trainloss
        total_test_loss += testloss

    avg_acc /= len(selected)
    avg_train_loss = total_train_loss / len(selected) if total_train_loss > 0 else None
    avg_test_loss = total_test_loss / len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    avg_adapt_lrs = accum_adapt_lrs / len(selected)

    nets_list = list(nets.values())
    return nets_list, avg_acc, avg_train_loss, avg_test_loss, avg_adapt_lrs

def local_train_net_per(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0 #This is for test accuracy
    total_train_loss = 0.0
    total_test_loss = 0.0
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            trainacc, testacc, trainloss, testloss = train_net_shakes(net_id, net, train_dl_local, test_dl_local, 
                n_epoch, args.lr, args.optimizer, args, device=device)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0

            if args.model in ['cnn-b']:
                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1, drop_last=True)
                elif args.noise_type == 'increasing':
                    noise_level = args.noise / (args.n_parties - 1) * net_id
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True, apply_noise=True)
                else:
                    noise_level = 0
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 2*args.batch_size, dataidxs_train, dataidxs_test, noise_level, drop_last=True)
        
            else:
                if args.noise_type == 'space':
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
                elif args.noise_type == 'increasing':
                    noise_level = args.noise / (args.n_parties - 1) * net_id
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, apply_noise=True)
                else:
                    noise_level = 0
                    train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
            
            trainacc, testacc, trainloss, testloss = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, 
                args.lr, args.optimizer, args, device=device)
        #
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        if trainloss is not None:
            total_train_loss += trainloss
        total_test_loss += testloss

    avg_acc /= len(selected)
    avg_train_loss = total_train_loss / len(selected) if total_train_loss > 0 else None
    avg_test_loss = total_test_loss / len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, avg_acc, avg_train_loss, avg_test_loss


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0 #This is for test accuracy
    total_train_loss = 0.0
    total_test_loss = 0.0
    n_epoch = args.epochs

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        net.to(device)

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
            trainacc, testacc, trainloss, testloss = train_net_shakes(net_id, net, train_dl_local, test_dl_local, 
            n_epoch, args.lr, args.optimizer, args, device=device, global_net=global_model, fedprox=True)
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)
            
            trainacc, testacc, trainloss, testloss = train_net(net_id, net, train_dl_local, test_dl_local, 
            n_epoch, args.lr, args.optimizer, args, device=device, global_net=global_model, fedprox=True)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        if trainloss is not None:
            total_train_loss += trainloss
        total_test_loss += testloss

    avg_acc /= len(selected)
    avg_train_loss = total_train_loss / len(selected) if total_train_loss > 0 else None
    avg_test_loss = total_test_loss / len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, avg_acc, avg_train_loss, avg_test_loss


def local_train_net_pfedMe(nets, selected, global_model, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu"):
    avg_acc = 0.0 #This is for test accuracy
    total_train_loss = 0.0
    total_test_loss = 0.0
    n_epoch = args.epochs
    update_dict = {}
    global_para = copy.deepcopy(global_model.state_dict())
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        regularized_local = copy.deepcopy(global_model)
        net.load_state_dict(global_para)
        net.to(device)

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
        
        else:
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
        
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        trainacc, testacc, trainloss, testloss, local_params = train_net_pfedMe(net_id, net, regularized_local, train_dl_local, test_dl_local, n_epoch, 
            args.lr, args.optimizer, args, device=device)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        if trainloss is not None:
            total_train_loss += trainloss
        total_test_loss += testloss
        update_dict[net_id] = local_params
    avg_acc /= len(selected)
    avg_train_loss = total_train_loss / len(selected) if total_train_loss > 0 else None
    avg_test_loss = total_test_loss / len(selected)
    
    return update_dict, avg_acc, avg_train_loss, avg_test_loss


def local_train_net_fedRod(nets, selected, p_head, args, net_dataidx_map_train, net_dataidx_map_test, logger=None, device="cpu", alpha=None):
    avg_acc = 0.0 #This is for test accuracy
    total_train_loss = 0.0
    total_test_loss = 0.0
    n_epoch = args.epochs
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)
        personal_head = p_head[net_id]
        sample_per_class = []

        if args.dataset == "shakespeare":
            train_dl_local = net_dataidx_map_train[net_id]
            test_dl_local = net_dataidx_map_test[net_id]
        
        else:
            sample_per_class = alpha[net_id]
            dataidxs_train = net_dataidx_map_train[net_id]
            dataidxs_test = net_dataidx_map_test[net_id]
            noise_level = args.noise
            if net_id == args.n_parties - 1:
                noise_level = 0
        
            if args.noise_type == 'space':
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * net_id
                train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        trainacc, testacc, trainloss, testloss = train_net_fedRod(net_id, net, personal_head, sample_per_class,
        train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, args, device=device)
        
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        if trainloss is not None:
            total_train_loss += trainloss
        total_test_loss += testloss
    avg_acc /= len(selected)
    avg_train_loss = total_train_loss / len(selected) if total_train_loss > 0 else None
    avg_test_loss = total_test_loss / len(selected)

    return p_head, avg_acc, avg_train_loss, avg_test_loss


def local_train_net_fedproto(nets, selected, args, net_dataidx_map_train, net_dataidx_map_test, global_protos, logger=None, device="cpu"):
    avg_acc = 0.0 #This is for test accuracy
    n_epoch = args.epochs
    uploaded_protos = []
    uploaded_ids = []
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)

        dataidxs_train = net_dataidx_map_train[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_divided_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs_train, dataidxs_test, noise_level)

        trainacc, testacc, trainloss, testloss, protos = train_net_fedproto(net_id, net, train_dl_local, test_dl_local, n_epoch, 
        global_protos, args.lr, args.optimizer, args, device=device)
        
        uploaded_protos.append(protos)
        uploaded_ids.append(net_id)

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        if trainloss is not None:
            total_train_loss += trainloss
        total_test_loss += testloss

    agg_protos_label = defaultdict(list)
    for local_protos in uploaded_protos:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    global_protos = copy.deepcopy(agg_protos_label)
    
    avg_acc /= len(selected)
    avg_train_loss = total_train_loss / len(selected) if total_train_loss > 0 else None
    avg_test_loss = total_test_loss / len(selected)

    return global_protos, avg_acc, avg_train_loss, avg_test_loss
