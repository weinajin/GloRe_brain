# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


'''
brats_iteartor: load brats file, output format np array of shape [C, W, H, D]
brats_transforms: data augmentation and turn input into tensor of shape [C, D, W, H]
'''
import os
import cv2
import time
import numpy as np

import torch
import torch.utils.data as data
import logging


import glob
import SimpleITK as sitk
import pandas as pd

def resample(image, spacing, size):
    # Create the reference image
    reference_origin = np.zeros(image.GetDimension())
    reference_direction = np.identity(image.GetDimension()).flatten()
    reference_image = sitk.Image(size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(spacing)
    reference_image.SetDirection(reference_direction)

    # Transform which maps from the reference_image to the current image (output-to-input)
    transform = sitk.AffineTransform(image.GetDimension())
    transform.SetMatrix(image.GetDirection())
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)

    # Modify the transformation to align the centers of the original and reference image
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    centering_transform = sitk.TranslationTransform(image.GetDimension())
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    # Using the linear interpolator
    image_rs = sitk.Resample(image, reference_image, transform, sitk.sitkLinear, 0.0)
    return image_rs


class BratsIter(data.Dataset):
    def __init__(self, csv_file, brats_path, brats_transform, shuffle=False):
        super(BratsIter, self).__init__()
        self.image_path = brats_path
        self.brats_transform = brats_transform
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['IDH'].notna()]  # get rid of data without gt label
        #         self.pairs = [row['BraTS19ID'], row['IDH'] for idx, row in pd.iterrows()]
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        logging.info("BratsIter:: iterator initialized (csv_file: '{:s}', num: {:d})".format(csv_file, len(self.df)))

    def __len__(self):
        return len(self.df)
    def  __getitem__(self, idx):
        '''
        Output: 4D torch.Tensor with dimensions (𝐶,𝐷,𝐻,𝑊)
        See: https://torchio.readthedocs.io/data/dataset.html
        '''
        label_dict = {'Mutant': 1, 'wt':0}
        label = label_dict[self.df.loc[idx, 'IDH']]
        bratsID = self.df.loc[idx, 'BraTS19ID']
        # read nii.gz, shape (W, H, C) (240, 240, 155)
        T1    = sitk.ReadImage(os.path.join(self.image_path, bratsID, bratsID+'_t1.nii.gz')) # (240, 240, 155)
        T1c   = sitk.ReadImage(os.path.join(self.image_path, bratsID, bratsID+'_t1ce.nii.gz'))
        T2    = sitk.ReadImage(os.path.join(self.image_path, bratsID, bratsID+'_t2.nii.gz'))
        FLAIR = sitk.ReadImage(os.path.join(self.image_path, bratsID, bratsID+'_flair.nii.gz'))

        # resample
        # x_size, y_size, z_size = T1.GetSize()
        # input_size = [224,224,8] #[128, 128, 128]
        # spacing = [x_size/input_size[0], y_size/input_size[1], z_size/input_size[2]]
        # T1    = resample(T1,    spacing=spacing, size=input_size)
        # T1c   = resample(T1c,   spacing=spacing, size=input_size)
        # T2    = resample(T2,    spacing=spacing, size=input_size)
        # FLAIR = resample(FLAIR, spacing=spacing, size=input_size)

        # convert to one batch of ndarray, shape will change from (W, H, D) to (D, H, W)
        # as GetArrayFromImage will transpose the shape
        # The input tensor must have 3 dimensions (D, H, W)
        # https://github.com/fepegar/torchio/blob/86a9641c6bec4bb5cc66db2628f205e7b1414bd2/torchio/data/image.py#L128
        T1    = sitk.GetArrayFromImage(T1).astype(np.float32)
        T1c   = sitk.GetArrayFromImage(T1c).astype(np.float32)
        T2    = sitk.GetArrayFromImage(T2).astype(np.float32)
        FLAIR = sitk.GetArrayFromImage(FLAIR).astype(np.float32)
        # size of [Channel, D, H, W], network receive input of: [C, D, W, H] 3,8,224,224
        image = np.stack((T1, T1c, T2, FLAIR), 0)

        # convert to torch tensor
        tensor = torch.from_numpy(image)

        # apply brats augmentation
        if self.brats_transform is not None:
           # if idx % 100 == 0:
               # self.brats_transform.set_random_state(seed=idx+int(time.time()))
            tensor = self.brats_transform(tensor)

        return tensor, label, bratsID 


if __name__ == "__main__":

    import pdb
    import time
    # from .iterator_factory_brats import read_brats_mean
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

    d_size, h_size, w_size = 155, 240, 240
    input_size = [7, 223,223]
    spacing = (d_size/input_size[0], h_size/input_size[1], w_size/input_size[2])
    training_transform = Compose([
        RescaleIntensity((0, 1)),  # so that there are no negative values for RandomMotion
        RandomMotion(),
        # HistogramStandardization({MRI: landmarks}),
        RandomBiasField(),
        ZNormalization(masking_method=ZNormalization.mean),
        RandomNoise(),
        ToCanonical(),
        Resample(spacing),
        # CropOrPad((48, 60, 48)),
        RandomFlip(axes=(0,)),
        OneOf({
            RandomAffine(): 0.8,
            RandomElasticDeformation(): 0.2,
        }),
    ])

    fold = 1
    data_root = '../../dld_data/brats2019/MICCAI_BraTS_2019_Data_Training/'

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Testing BratsIter without transformer [not torch wapper]")


    # has memory (no metter random or not, it will trival all none overlapped clips)
    train_dataset = BratsIter(csv_file=os.path.join(data_root, 'IDH_label', 'train_fold_{}.csv'.format(fold)),
                         brats_path = os.path.join(data_root, 'all'),
                         brats_transform=training_transform,
                         shuffle=True)

    # Mean, Std, Max = read_data_mean(fold, trainset)
    # normalize = transforms.Normalize(Mean,Std)


    for i in range(1, 2):
        img, lab, bratsID = train_dataset.__getitem__(i)
        logging.info("{}: {}".format(i, img.shape))

    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1, shuffle=True,
                                               num_workers=12, pin_memory=True)

    logging.info("Start iter")
    tic = time.time()
    for i, (img, lab, bratsID) in enumerate(train_loader):
        t = time.time() - tic
        logging.info("{} samples/sec. \t img.shape = {}, label = {}, bratsID = {}.".format(float(i+1)/t, img.shape, lab, bratsID))
        if i == 1:
            break

    
    import matplotlib.pyplot as plt
     
    for vid in range(2):

        img, lab, id = train_dataset.__getitem__(1)
        img = np.clip(img, 0., 1.)
        logging.info(img.shape)
        for i in range(0, 5):
            plt.imshow(img.numpy()[:,i,:,:].transpose(1,2,0))
            plt.draw()
            plt.pause(0.2)
    
    plt.pause(1)
    plt.savefig('dataloader.png')
