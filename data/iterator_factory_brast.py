# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import torch

from .brast_iterator import BratsIter

def read_brats_mean(fold, trainset):
    # Mean = torch.from_numpy(np.array([359.7985, 442.3089, 442.4439, 315.7795]).astype(np.float32))
    # Std  = torch.from_numpy(np.array([531.5512, 641.3847, 670.7475, 464.5106]).astype(np.float32))
    # '''
    pickle_name = os.path.join(opt.log_path, 'mean_fold_{}'.format(fold))
    try:
        Mean, Std, Max = pickle.load(open(pickle_name, "rb"))
    except OSError as e:
        Mean = torch.zeros(4)
        Std = torch.zeros(4)
        Max = torch.zeros(4)
        kkk = 0
        for i in range(len(trainset)):
            I, L = trainset[i]
            C,D,W,H = I.size()
            Mean += I.view(C,-1).mean(1)
            Std += I.view(C,-1).std(1)
            MM = torch.max(I.view(C,-1),dim=1)[0]
            for i in range(4):
                if MM[i] > Max[i]:
                    Max[i] = MM[i]
            kkk += 1
            print(kkk, end=" ")
        Mean /= len(trainset)
        Std  /= len(trainset)
        pickle.dump([Mean, Std, Max], open(pickle_name, "wb"))
    print('\n mean for fold {}: '.format(fold)), print(Mean.numpy())
    print('std: '), print(Std.numpy())
    print('max: '), print(Max.numpy())
    return Mean, Std, Max


def get_brats(data_root='../../dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/all',
                 fold = 0,
                 seed=torch.distributed.get_rank() if torch.distributed._initialized else 0,
                 **kwargs):
    """ data iter for brats
    """
    logging.debug("BratsIter:: fold = {}, seed = {}".format(fold, seed))
    Mean, Std, Max = read_data_mean(fold, trainset)

    normalize = transforms.Normalize(mean=Mean, std=Std)

    train_sampler = sampler.RandomSampling(num=clip_length,
                                           interval=train_interval,
                                           speed=[1.0, 1.0],
                                           seed=(seed+0))
    train = BratsIter(brats_prefix=os.path.join(data_root, 'raw', 'data', 'train_avi-288p'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'brats_train_avi.txt'),
                      sampler=train_sampler,
                      force_color=True,
                      brats_transform=transforms.Compose([
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
    val   = BratsIter(brats_prefix=os.path.join(data_root, 'raw', 'data', 'val_avi-288p'),
                      txt_list=os.path.join(data_root, 'raw', 'list_cvt', 'brats_val_avi.txt'),
                      sampler=val_sampler,
                      force_color=True,
                      brats_transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.CenterCrop((224, 224)),
                                         transforms.ToTensor(),
                                         normalize,
                                      ]),
                      name='test',
                      shuffle_list_seed=(seed+3),
                      )
    return (train, val)



def create(name, fold, batch_size, num_workers=8, **kwargs):

    if name.upper() == 'BRATS':
        train, val = get_brats(**kwargs)
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
