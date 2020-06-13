# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging

import torch

from . import video_sampler as sampler
from . import video_transforms as transforms
from .video_iterator import VideoIter

def get_kinetics(data_root='./dataset/Kinetics',
                 clip_length=8,
                 train_interval=2,
                 val_interval=2,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
                 **kwargs):
    """ data iter for kinetics
    """
    logging.debug("VideoIter:: clip_length = {}, interval = [train: {}, val: {}], seed = {}".format( \
                clip_length, train_interval, val_interval, seed))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'train_avi-288p'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_train_avi.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.RandomScale(make_square=False,
                                                                aspect_ratio=[.8, 1./.8],
                                                                slen=[224, 340]),
                                         transforms.RandomCrop((224, 224)), # insert a resize if needed
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomHLS(vars=[15, 35, 25]), # too slow
                                         transforms.PixelJitter(vars=[-20, 20]), 
                                         transforms.ToTensor(),
                                         normalize,
                                      ],
                                      aug_seed=(seed+1)),
                      name='train',
                      shuffle_list_seed=(seed+2),
                      )

    val_sampler   = sampler.EvenlySampling(num=clip_length,
                                           interval=val_interval,
                                           num_times=1)
    val   = VideoIter(video_prefix=os.path.join(data_root, 'raw', 'data', 'val_avi-288p'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'kinetics_val_avi.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      video_transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      shuffle_list_seed=(seed+3),
                      )
    return (train, val)



def creat(name, batch_size, num_workers=50, **kwargs):

    if name.upper() == 'KINETICS':
        train, val = get_kinetics(**kwargs)
    else:
        assert NotImplementedError("iter {} not found".format(name))

    train_loader = torch.utils.data.DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(val,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    return (train_loader, val_loader)
