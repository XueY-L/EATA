'''
CUDA_VISIBLE_DEVICES=1 python3 main_domainnet126.py
'''

from collections import OrderedDict
from logging import debug
import os
import time
import argparse
import json
import random
import math

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data

import torch    
import torch.nn.functional as F
import torchvision.models as tmodels
import torchvision.transforms as transforms
import numpy as np

import tent
import eata
from image_list import ImageList

import models.Res as Resnet

from robustbench.data import load_imagenetc
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)
    return top1.avg, top5.avg


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/home/yxue/datasets/ILSVRC', help='path to dataset')
    parser.add_argument('--data_corruption', default='/home/yxue/datasets/DomainNet-126', help='path to corruption dataset')
    parser.add_argument('--output', default='etta_exps/camera_ready_debugs', help='the output directory of this experiment')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=50, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=False, type=bool, help='if shuffle the test set.')

    parser.add_argument('--fisher_clip_by_norm', type=float, default=10.0, help='Clip fisher before it is too large')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='', type=str, help='corruption type of test(val) set.')
    parser.add_argument('--rotation', default=False, type=bool, help='if use the rotation ssl task for training (this is TTTs dataloader).')

    # model name, support resnets
    parser.add_argument('--arch', default='resnet50', type=str, help='the default model architecture')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(126)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    

    # overall experimental settings
    parser.add_argument('--exp_type', default='continual', type=str, help='continual or each_shift_reset')
    # 'cotinual' means the model parameters will never be reset, also called online adaptation; 
    # 'each_shift_reset' means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will be reset.
    parser.add_argument('--algorithm', default='eata', type=str, help='eata or eta or tent')  # eta不加权重正则

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    source_domain = 'sketch'
    target_domains = ['clipart', 'painting']

    class ImageNormalizer(nn.Module):
        def __init__(self, mean, std):
            super(ImageNormalizer, self).__init__()

            self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
            self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

        def forward(self, inp):
            if isinstance(inp, tuple):
                return ((inp[0] - self.mean) / self.std, inp[1])
            else:
                return (inp - self.mean) / self.std
    
    subnet = nn.Sequential(
        OrderedDict([
            ('normalize', ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
            ('model', tmodels.resnet50(num_classes=126)),
        ])
    ).cuda()
    subnet.model.load_state_dict(torch.load(f'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_{source_domain}__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth')['net'])

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    logger = get_logger(name="project", output_directory=args.output, log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt", debug=False)
    
    logger.info(args)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    if args.algorithm == 'eata':
        # compute fisher informatrix
        label_file = os.path.join(args.data_corruption, f"{source_domain}_list.txt")
        test_dataset = ImageList(args.data_corruption, label_file, transform=test_transform)
        fisher_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=False)

        subnet = eata.configure_model(subnet)
        params, param_names = eata.collect_params(subnet)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets, _) in enumerate(fisher_loader, start=1):      
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)
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

        optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)
        adapt_model = eata.EATA(subnet, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
    else:
        assert False, NotImplementedError

    for d in target_domains:
        label_file = os.path.join(args.data_corruption, f"{d}_list.txt")
        test_dataset = ImageList(args.data_corruption, label_file, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=False)

        top1, top5 = validate(test_loader, adapt_model, None, args, mode='eval')
        logger.info(f"Under shift type {d} After {args.algorithm} Top-1 Accuracy: {top1:.5f} and Top-5 Accuracy: {top5:.5f}")
        if args.algorithm in ['eata', 'eta']:
            logger.info(f"num of reliable samples is {adapt_model.num_samples_update_1}, num of reliable+non-redundant samples is {adapt_model.num_samples_update_2}")
            adapt_model.num_samples_update_1, adapt_model.num_samples_update_2 = 0, 0


    