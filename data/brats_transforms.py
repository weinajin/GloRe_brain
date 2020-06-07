import numpy as np
import random, itertools

'''
Data augmentation on-the-fly
For Brats data, use affine transformation. Ref: https://www.frontiersin.org/articles/10.3389/fncom.2019.00083/full
Example code https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
'''
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms, aug_seed=0):
        self.transforms = transforms
        self.set_random_state(aug_seed)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    def set_random_state(self, seed=None):
        for i, t in enumerate(self.transforms):
            t.set_random_state(seed=(seed+i))

class Transform(object):
    """basse class for all transformation"""
    def set_random_state(self, seed=None):
        self.rng = np.random.RandomState(seed)


####################################
# Customized Transformations
####################################

class ToTensor(Transform):
    """
    Converts a numpy.ndarray (W x H x (D x C)) to a torch.FloatTensor of shape (C x D x W x H).
    """
    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            H, W, _ = img.shape
            # handle numpy array
            img = torch.from_numpy(img.reshape((H,W,-1,self.dim)).transpose((3, 2, 0, 1)))
            # backward compatibility
            return img.float().

def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.
    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)
    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.combinations_with_replacement(range(4), 2), range(2), range(2), range(2), range(2)))

def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))

def permute_data(data_in, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, z, y, x).
    Input key is a tuple: ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)
    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data_in)
    # (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key
    (__, rotate_z), flip_x, flip_y, flip_z, transpose = key

    # if rotate_y != 0:
    #     data = np.rot90(data, rotate_y, axes=(1, 3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_z:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_x:
        data = data[:, :, :, ::-1]
    if transpose:
        data = np.transpose(data, (0,1,3,2)) # transpose x, y
    return data.copy()

def random_permutation_x_y(x_data):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :return: the permuted data
    """
    key = random_permutation_key()
    return permute_data(x_data, key)

def augment_data(x_data, augment=None):
    # assuming a batch will be coming with first dimension = batch size. So we permute for all examples in this batch
    return random_permutation_x_y(x_data)
    # if len(x_data.shape) < 5:
    #     x_data = np.expand_dims(x_data, 0)
    # if 'permute' in augment:
    #     for curr_eg in range(x_data.shape[0]):
    #         x_data[curr_eg,] = random_permutation_x_y(x_data[curr_eg,])
    # if 'add_noise' in augment:
    #     x_data = add_noise(x_data)
    # if 'add_blur' in augment:
    #     x_data = add_blur(x_data)
    # if 'affine' in augment:
    #     x_data = translate_data(x_data)
    #     x_data = scale_data(x_data)
    #     x_data = shear_data(x_data)
    # return x_data
