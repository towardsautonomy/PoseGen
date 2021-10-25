# import packages
import os
import sys
import copy
import numpy as np
import scipy
import scipy.io
from dataset import Dataset

# add path to sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(os.path.abspath('src/utils'))
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Dataset Class
class StanfordCarsDataset(Dataset):

    def __init__(self, dataroot, resize_dim, 
                       transforms=None,
                       retrieve_by_id=False, 
                       object_id=None, 
                       shuffle=True, 
                       verbose=True):
        """
        Args:
            dataroot (string): Root Directory of Stanford Cars dataset.
            resize_dim (tuple(w, h)): Dimension to resize the images to.
            retrieve_by_id (bool): Whether or not to retrieve images by their object id.
            object_id (int): Object ID to retrieve data for - only applicable when `retrieve_by_id` is set.
            shuffle (bool): Whether or not to shuffle the dataset.
            verbose (bool): Whether or not to print additional information.
        """
        self.dataroot = dataroot
        self.resize_dim = resize_dim
        self.transforms = transforms
        self.retrieve_by_id = retrieve_by_id
        self.object_id = object_id
        self.shuffle = shuffle
        self.verbose = verbose

        metadata_filename = os.path.join(dataroot, 'car_devkit/devkit/cars_meta.mat')
        annotations_filename = os.path.join(dataroot, 'car_devkit/devkit/cars_train_annos.mat')
        train_dir = os.path.join(dataroot, 'cars_train')
        test_dir = os.path.join(dataroot, 'cars_test')

        # read metadata
        cars_metadata = {}
        mat = scipy.io.loadmat(metadata_filename)
        for i, car_model in enumerate(mat['class_names'][0]):
            cars_metadata[i+1] = car_model[0]

        # pairs of object type and filenames
        self.id_filaneme_pairs = []

        # dictionary of object type and description
        self.id_description_dict = cars_metadata.copy()

        # read annotations
        self.annotations = {}
        mat = scipy.io.loadmat(annotations_filename)
        for ann in mat['annotations'][0]:
            filename = os.path.join(train_dir, ann[-1][0])
            if os.path.exists(filename):
                car_type_id = int(ann[-2][0][0])
                if car_type_id in self.annotations.keys():
                    self.annotations[car_type_id].append(filename)
                else:
                    self.annotations[car_type_id] = [filename]
                self.id_filaneme_pairs.append({car_type_id: filename})

        # list of car type ids
        self.car_type_ids = list(self.annotations.keys())

        # shuffle
        if shuffle == True:
            np.random.shuffle(self.id_filaneme_pairs)

        # print stats
        if self.verbose:
            print('Number of samples found in the dataset: {}'.format(self.__len__()))

    # method to get length of data
    def __len__(self):
        if self.retrieve_by_id==False:
            return len(self.id_filaneme_pairs)
        else:
            assert(self.object_id in self.car_type_ids)
            return len(self.annotations[self.object_id])

    # method to get resize shape
    def get_resize_dim(self):
        return self.resize_dim

    # method to get transforms functions
    def get_transforms(self):
        return self.transforms

    # method to get a list of object type
    def get_object_type_ids(self):
        return np.array(self.car_type_ids)

    # method to get a list of [object type<->filename] pairs
    def get_id_filename_pairs(self):
        return self.id_filaneme_pairs

    # method to get a list of filenames corresponding to the object id
    def get_filenames(self):
        assert(self.object_id in self.get_object_type_ids())
        return self.annotations[self.object_id]

    # method to get a dictionary of {object type: object_description} pairs
    def object_id_description_dict(self):
        return self.id_description_dict

# main function
if __name__ == '__main__':
    import cv2

    ## dataset object
    # dataset = StanfordCarsDataset( dataroot='/floppy/datasets/Stanford', 
    #                                resize_dim=(256,256), 
    #                                retrieve_by_id=True, 
    #                                object_id=1, 
    #                                verbose=True)
    dataset = StanfordCarsDataset( dataroot='/floppy/datasets/Stanford', 
                                   resize_dim=(256,256),
                                   verbose=True)

    # display samples
    for sample in dataset:
        cv2.namedWindow(sample['object_description'])
        img_bgr = cv2.cvtColor(np.array(sample['image']), cv2.COLOR_RGB2BGR)
        cv2.imshow(sample['object_description'], img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    