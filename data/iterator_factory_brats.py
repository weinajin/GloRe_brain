# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import torch
import pickle
from .brats_iterator import BratsIter
from . import brats_transforms as transforms

from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)

def read_brats_mean(fold, data_root):
    # Mean = torch.from_numpy(np.array([359.7985, 442.3089, 442.4439, 315.7795]).astype(np.float32))
    # Std  = torch.from_numpy(np.array([531.5512, 641.3847, 670.7475, 464.5106]).astype(np.float32))
    # '''
    csv_file = os.path.join(data_root, 'IDH_label', 'train_fold_{}.csv'.format(fold))
    brats_path = os.path.join(data_root, 'all')
    mean_file = os.path.join(data_root, 'IDH_label', 'mean_fold_{}'.format(fold))
    try:
        Mean, Std, Max = pickle.load(open(mean_file, "rb"))
    except OSError as e:
        Mean = torch.zeros(4)
        Std = torch.zeros(4)
        Max = torch.zeros(4)
        kkk = 0
        trainset = BratsIter(csv_file=csv_file,
                      brats_path = brats_path,
                      brats_transform= None,
                      shuffle=False)
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
            logging.INFO(kkk, end=" ")
        Mean /= len(trainset)
        Std  /= len(trainset)
        pickle.dump([Mean, Std, Max], open(mean_file, "wb"))
    logging.info('Mean for fold {}: {}'.format(fold, Mean.numpy()))
    logging.info('Std: {}'.format(Std.numpy()))
    logging.info('Max: {}'.format(Max.numpy()))
    return Mean, Std, Max


def get_brats(data_root='/scratch/weina/dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/',
              fold = 1,
              seed=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
              **kwargs):
    """ data iter for brats
    """
    logging.debug("BratsIter:: fold = {}, seed = {}".format(fold, seed))
    # args for transforms
    d_size, h_size, w_size = 155, 240, 240
    input_size = [7, 223,223]
    spacing = (d_size/input_size[0], h_size/input_size[1], w_size/input_size[2])
    Mean, Std, Max = read_brats_mean(fold, data_root)
    normalize = transforms.Normalize(mean=Mean, std=Std)
    training_transform = Compose([
        # RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        # RandomMotion(),
        # HistogramStandardization({MRI: landmarks}),
        RandomBiasField(),
        # ZNormalization(masking_method=ZNormalization.mean),
        RandomNoise(),
        ToCanonical(),
        Resample(spacing),
        # CropOrPad((48, 60, 48)),
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
        normalize
    ])
    val_transform = Compose([
        Resample(spacing),
        normalize
    ])

    train = BratsIter(csv_file=os.path.join(data_root, 'IDH_label', 'train_fold_{}.csv'.format(fold)),
                      brats_path = os.path.join(data_root, 'all'),
                      brats_transform=training_transform,
                      shuffle=True)

    val   = BratsIter(csv_file=os.path.join(data_root, 'IDH_label', 'val_fold_{}.csv'.format(fold)),
                      brats_path = os.path.join(data_root, 'all'),
                      brats_transform=val_transform,
                      shuffle=False)
    return train, val



def create(name, fold, batch_size, num_workers=8, **kwargs):

    if name.upper() == 'BRATS_IDH':
        train, val = get_brats(fold = fold, **kwargs)
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

if __name__ == "__main__":
    import time
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('start test iterator_factor_brats')
    train, val = get_brats()
    for i in range(1, 2):
        img, lab, bratsID = val.__getitem__(i)
        logging.info('max: {}, min: {}, mean: {}, std: {}'.format(img.max(), img.min(), img.mean(), img.std()))
        logging.info("{}: {}".format(i, img.shape))
    train_loader = torch.utils.data.DataLoader(val,
                                               batch_size=1, shuffle=True,
                                               num_workers=12, pin_memory=True)
    logging.info("Start iter")
    tic = time.time()
    for i, (img, lab, bratsID) in enumerate(train_loader):
        t = time.time() - tic
        logging.info("{} samples/sec. \t img.shape = {}, label = {}, bratsID = {}.".format(float(i+1)/t, img.shape, lab, bratsID))
        if i == 1:
            break
