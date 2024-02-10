'''
CUDA_VISIBLE_DEVICES=3 python3 main_cifar100c.py
'''

from logging import debug
import os
import time
import argparse
import json
import random
import math

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data, prepare_cifar100_test_data

import torch    
import torch.nn.functional as F
import torchvision.models as tmodels
import numpy as np

import tent
import eata

import models.Res as Resnet

from robustbench.data import load_cifar100c, load_imagenetc
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/home/yxue/datasets/ILSVRC', help='path to dataset')
    parser.add_argument('--data_corruption', default='/home/yxue/datasets', help='path to corruption dataset')
    parser.add_argument('--output', default='etta_exps/camera_ready_debugs', help='the output directory of this experiment')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=False, type=bool, help='if shuffle the test set.')

    parser.add_argument('--fisher_clip_by_norm', type=float, default=10.0, help='Clip fisher before it is too large')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    parser.add_argument('--rotation', default=False, type=bool, help='if use the rotation ssl task for training (this is TTTs dataloader).')

    # model name, support resnets
    parser.add_argument('--arch', default='resnet50', type=str, help='the default model architecture')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    

    # overall experimental settings
    parser.add_argument('--exp_type', default='continual', type=str, help='continual or each_shift_reset') 
    # 'cotinual' means the model parameters will never be reset, also called online adaptation; 
    # 'each_shift_reset' means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will be reset.
    parser.add_argument('--algorithm', default='eata', type=str, help='eata or eta or tent')  # eta不加权重正则

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    args.data = '/home/yxue/datasets/cifar100/cifar-100-python'
    args.data_corruption = '/home/yxue/datasets/CIFAR-100-C'
    args.batch_size = 200
    args.e_margin = math.log(100)*0.40
    args.d_margin = 0.4
    args.fisher_alpha = 1

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    subnet = load_model('Hendrycks2020AugMix_ResNeXt', './ckpt', 'cifar100', ThreatModel.corruptions).cuda()
    subnet.load_state_dict(torch.load('/home/yxue/model_fusion_tta/cifar/checkpoint/ckpt_cifar100_[\'jpeg_compression\']_[1].pt')['model'])

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    logger = get_logger(name="project", output_directory=args.output, log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt", debug=False)
    
    # ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    logger.info(args)

    if args.exp_type == 'continual':  # 在每个域推理完成后，测一下original情况
        # common_corruptions = [[item, 'original'] for item in common_corruptions]
        # common_corruptions = [subitem for item in common_corruptions for subitem in item]  #在common_corruptions的每项后面都加了个'original'
        pass
    elif args.exp_type == 'each_shift_reset':
        print("continue")
    else:
        assert False, NotImplementedError
    logger.info(common_corruptions)
    
    if args.algorithm == 'eata':
        # compute fisher informatrix
        args.corruption = 'original'

        fisher_loader = prepare_cifar100_test_data(args)

        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):
            if iter_ == args.fisher_size // args.batch_size: break  # 用少量样本算fisher
            if args.gpu is not None:
                images = images.cuda(non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(non_blocking=True)
            outputs = subnet(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in subnet.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("compute fisher matrices finished")
        del ewc_optimizer

        optimizer = torch.optim.SGD(params, 0.005, momentum=0.9)
        adapt_model = eata.EATA(subnet, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
    else:
        assert False, NotImplementedError

    for corrupt in common_corruptions:
        x_test, y_test = load_cifar100c(10000, 5, '/home/yxue/datasets', False, [corrupt])
        x_test, y_test = x_test.cuda(), y_test.cuda()

        acc = 0.
        n_batches = math.ceil(x_test.shape[0] / args.batch_size)
        with torch.no_grad():
            for counter in range(n_batches):
                x_curr = x_test[counter * args.batch_size:(counter + 1) *
                        args.batch_size].cuda()
                y_curr = y_test[counter * args.batch_size:(counter + 1) *
                        args.batch_size].cuda()

                output = adapt_model(x_curr)
                acc += (output.max(1)[1] == y_curr).float().sum()
        print(f'{acc.item() / x_test.shape[0]:.4}', end=' ')
    