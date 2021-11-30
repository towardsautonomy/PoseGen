# import packages
import os
import sys

import numpy as np
from PIL import Image, ImageOps
from copy import deepcopy
import torch
import torchvision.transforms as transforms

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('src/utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

# import utilities
from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# dataset class from which each dataset inherits
class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        pass

    # method to get length of data
    def __len__(self):
        raise NotImplementedError

    # method to get each item
    def __getitem__(self, idx, object_id=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = None
        if object_id == None:
            id_filename_pairs = self.get_id_filename_pairs()
            assert(idx < len(id_filename_pairs))
            id_filename_pair = id_filename_pairs[idx]
            obj_type_id = list(id_filename_pair.keys())[0]
            filename = list(id_filename_pair.values())[0]
            # object image
            obj_image = Image.open(filename['image']).resize(self.get_resize_dim())
            # check for grayscale image
            if obj_image.mode == 'L':
                obj_image = ImageOps.colorize(obj_image, black ="black", white ="white")
            # perform transforms
            if self.get_transforms() is not None:
                obj_image = self.get_transforms()(obj_image)

            # target image
            target_idx = np.random.randint(0, len(id_filename_pairs))
            target_id_filename_pair = id_filename_pairs[target_idx]
            target_filename = list(target_id_filename_pair.values())[0]
            target_image = Image.open(target_filename['image']).resize(self.get_resize_dim())
            # check for grayscale image
            if target_image.mode == 'L':
                target_image = ImageOps.colorize(target_image, black ="black", white ="white")
            # perform transforms
            if self.get_transforms() is not None:
                target_image = self.get_transforms()(target_image)

            sil_image = Image.open(target_filename['mask']).resize(self.get_resize_dim())
            # check for grayscale image
            if sil_image.mode == 'L':
                sil_image = ImageOps.colorize(sil_image, black ="black", white ="white")
            # perform transforms
            if self.get_transforms() is not None:
                sil_image = self.get_transforms()(sil_image)

            sample = { 'ref_image': obj_image,
                       'target_image': target_image,
                       'sil_image': sil_image,
                       'object_type_id': obj_type_id,
                       'object_description': self.object_id_description_dict()[obj_type_id]
                     }
        else:
            if object_id in self.get_object_type_ids():
                filenames = self.get_filenames(object_id)
                assert(idx < len(filenames))
                # object image
                obj_image = Image.open(filenames[idx]['image']).resize(self.get_resize_dim())
                # check for grayscale image
                if obj_image.mode == 'L':
                    obj_image = ImageOps.colorize(obj_image, black ="black", white ="white")
                # perform transforms
                if self.get_transforms() is not None:
                    obj_image = self.get_transforms()(obj_image)

                # target image
                target_idx = np.random.randint(0, len(filenames))
                target_id_filename_pair = filenames[target_idx]
                target_filename = list(target_id_filename_pair.values())[0]
                target_image = Image.open(target_filename['image']).resize(self.get_resize_dim())
                # check for grayscale image
                if target_image.mode == 'L':
                    target_image = ImageOps.colorize(target_image, black ="black", white ="white")
                # perform transforms
                if self.get_transforms() is not None:
                    target_image = self.get_transforms()(target_image)

                # silhouette image
                sil_image = Image.open(filenames[target_idx]['mask']).resize(self.get_resize_dim())
                # check for grayscale image
                if sil_image.mode == 'L':
                    sil_image = ImageOps.colorize(sil_image, black ="black", white ="white")
                # perform transforms
                if self.get_transforms() is not None:
                    sil_image = self.get_transforms()(sil_image)

                sample = { 'ref_image': obj_image,
                           'target_image': target_image,
                           'object_type_id': object_id,
                           'object_description': self.object_id_description_dict()[object_id]
                        }
        return sample

    # method to get resize shape
    def get_resize_dim(self):
        '''
        Returns the dimensions of image size to be resized to (w, h)
        '''
        raise NotImplementedError

    # method to get transforms functions
    def get_transforms(self):
        '''
        Returns a list of transforms function
        '''
        raise NotImplementedError

    # method to get a list of object type
    def get_object_type_ids(self):
        '''
        Returns a numpy array of object type ids
        '''
        raise NotImplementedError

    # method to get annotations
    def get_annotations(self):
        '''
        Returns a dictionary of {object_type_id: [filenames]}
        '''
        raise NotImplementedError

    # method to get a list of filenames corresponding to the object id
    def get_filenames(self):
        '''
        Returns a list of filenames
        '''
        raise NotImplementedError

    # method to get a list of background image filenames
    def get_background_filenames(self):
        raise NotImplementedError

    # method to get a list of silhouette image filenames
    def get_silhouette_filenames(self):
        raise NotImplementedError

    # method to get a list of [object type<->filename] pairs
    def get_id_filename_pairs(self):
        '''
        Returns a list of [{object_type_id1: filename1}, {object_type_id2: filename2}, ...]
        '''
        raise NotImplementedError

    # method to get a dictionary of {object type: object_description} pairs
    def object_id_description_dict(self):
        '''
        Returns a dictionary of {object_type1: object_description1, object_type2: object_description2} 
        '''
        raise NotImplementedError