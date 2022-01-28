# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""Code for getting the data loaders."""

import numpy as np
from PIL import Image
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import urllib
from lmdb_datasets import LMDBDataset
from thirdparty.lsun import LSUN

def get_loaders(args):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args)

def get_loaders_eval(dataset, args):
    if dataset.startswith('minecraft'):
        num_classes = 1
        resize = 64
        train_transform, valid_transform = _data_transforms_minecraft(resize)
        train_data = LMDBDataset(root=args.data, name='minecraft', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=args.data, name='minecraft', train=False, transform=valid_transform)
    else:
        raise NotImplementedError

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    num_workers_train = 8
    num_workers_valid = 1
    if (args.OS == "windows" or args.OS == "win"):
        num_workers_train = 0
        num_workers_valid = 0

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=num_workers_train, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=num_workers_valid, drop_last=False)

    return train_queue, valid_queue, num_classes

def _data_transforms_minecraft(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform
