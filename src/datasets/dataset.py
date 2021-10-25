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
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = None
        if self.retrieve_by_id==False:
            id_filename_pairs = self.get_id_filename_pairs()
            assert(idx < len(id_filename_pairs))
            id_filename_pair = id_filename_pairs[idx]
            obj_type_id = list(id_filename_pair.keys())[0]
            filename = list(id_filename_pair.values())[0]
            image = Image.open(filename).resize(self.get_resize_dim())
            # check for grayscale image
            if image.mode == 'L':
                image = ImageOps.colorize(image, black ="blue", white ="white")
            # perform transforms
            if self.get_transforms() is not None:
                image = self.get_transforms()(image)
            sample = { 'image': image,
                       'object_type_id': obj_type_id,
                       'object_description': self.object_id_description_dict()[obj_type_id]
                     }
        else:
            if self.object_id in self.get_object_type_ids():
                filenames = self.get_filenames(self.object_id)
                assert(idx < len(filenames))
                image = Image.open(filenames[idx]).resize(self.get_resize_dim())
                # check for grayscale image
                if image.mode == 'L':
                    image = ImageOps.colorize(image, black ="blue", white ="white")
                # perform transforms
                if self.get_transforms() is not None:
                    image = self.get_transforms()(image)
                sample = { 'image': image,
                           'object_type_id': self.object_id,
                           'object_description': self.object_id_description_dict()[self.object_id]
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