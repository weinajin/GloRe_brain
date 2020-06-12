import numpy as np
import random, itertools
from torchio import Transform, DATA

'''
Data augmentation on-the-fly
For Brats data, use affine transformation. Ref: https://www.frontiersin.org/articles/10.3389/fncom.2019.00083/full
Example code https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
'''
# class Compose(object):
#     """Composes several transforms together.
#
#     Args:
#         transforms (List[Transform]): list of transforms to compose.
#
#     Example:
#         >>> transforms.Compose([
#         >>>     transforms.CenterCrop(10),
#         >>>     transforms.ToTensor(),
#         >>> ])
#     """
#     def __init__(self, transforms, aug_seed=0):
#         self.transforms = transforms
#         self.set_random_state(aug_seed)
#
#     def __call__(self, data):
#         for t in self.transforms:
#             data = t(data)
#         return data
#
#     def __repr__(self):
#         format_string = self.__class__.__name__ + '('
#         for t in self.transforms:
#             format_string += '\n'
#             format_string += '    {0}'.format(t)
#         format_string += '\n)'
#         return format_string
#
#     def set_random_state(self, seed=None):
#         for i, t in enumerate(self.transforms):
#             t.set_random_state(seed=(seed+i))
#
# class Transform(object):
#     """basse class for all transformation"""
#     def set_random_state(self, seed=None):
#         self.rng = np.random.RandomState(seed)
#

####################################
# Customized Transformations
####################################

# class ToTensor(Transform):
#     """
#     Converts a numpy.ndarray (W x H x (D x C)) to a torch.FloatTensor of shape (C x D x W x H).
#     """
#     def __init__(self, dim=3):
#         self.dim = dim
#
#     def __call__(self, img):
#         if isinstance(img, np.ndarray):
#             H, W, _ = img.shape
#             # handle numpy array
#             img = torch.from_numpy(img.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1)))
#             # backward compatibility
#             return img.float()

class Normalize(Transform):
    '''
    Normalize a torch.FloatTensor of (C, D, H, W) to the same shape. with its mean and std are unscaled (300+).
    Given mean and std of size (C, ),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.probability = 1.0

    def apply_transform(self, sample):
        for image_dict, mn, std in zip(sample.get_images_dict().values(), self.mean, self.std):
            # max: 1488.2520751953125, min: -0.3330691158771515, mean: 29.41769790649414, std: 94.70357513427734
            normalize = image_dict[DATA].sub_(mn).div_(std)
            # max: 4.941013813018799, min: -0.4125528335571289, mean: -0.30555957555770874, std: 0.340593159198761
            image_dict[DATA] = normalize
        return sample

