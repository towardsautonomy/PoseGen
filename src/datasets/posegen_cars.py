# import packages
import os
import sys
import copy
import glob
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
class PoseGenCarsDataset(Dataset):

    def __init__(self, dataroot, 
                       resize_dim, 
                       transforms=None,
                       object_id=None, 
                       shuffle=True,
                       image_ext='JPEG',
                       verbose=True):
        """
        Args:
            dataroot (string): Root Directory of PoseGen Cars dataset.
            resize_dim (tuple(w, h)): Dimension to resize the images to.
            object_id (int): Object ID to retrieve data for - set to None if all classes are needed.
            shuffle (bool): Whether or not to shuffle the dataset.
            verbose (bool): Whether or not to print additional information.
        """
        self.dataroot = dataroot
        self.resize_dim = resize_dim
        self.transforms = transforms
        self.object_id = object_id
        self.shuffle = shuffle
        self.verbose = verbose

        # pairs of object type and filenames
        self.car_type_ids = []
        self.id_filaneme_pairs = []
        # dictionary of object type and description
        self.id_description_dict = {}
        # read annotations
        self.annotations = {}
        # get list of object types defined by dir names
        car_models = os.listdir(self.dataroot)
        # build list of objects
        for car_type_id, car_model in enumerate(car_models):
            if os.path.isdir(os.path.join(self.dataroot, car_model)):
                places = os.listdir(os.path.join(self.dataroot, car_model))
                for place in places:
                    self.car_type_ids.append(car_type_id)
                    img_filenames = glob.glob(os.path.join(self.dataroot, car_model, place, 'image/*.{}'.format(image_ext)))
                    for img_filename in img_filenames:
                        sil_filename = img_filename.replace('image', 'mask')
                        if os.path.exists(img_filename) and os.path.exists(sil_filename):
                            if car_type_id in self.annotations.keys():
                                self.annotations[car_type_id].append(img_filename)
                            else:
                                self.annotations[car_type_id] = [img_filename]
                                self.id_description_dict[car_type_id] = car_model
                            self.id_filaneme_pairs.append({car_type_id: {'image': img_filename, 'mask': sil_filename}})

        # shuffle
        if shuffle == True:
            np.random.shuffle(self.id_filaneme_pairs)

        # print stats
        if self.verbose:
            print('Number of samples found in the dataset: {}'.format(self.__len__()))

    # method to get length of data
    def __len__(self, object_id=None):
        return len(self.id_filaneme_pairs)

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
    def get_filenames(self, object_id=None):
        assert(object_id in self.get_object_type_ids())
        return self.annotations[object_id]

    # method to get a dictionary of {object type: object_description} pairs
    def object_id_description_dict(self):
        return self.id_description_dict

# main function
if __name__ == '__main__':
    import cv2

    ## dataset object
    dataset = PoseGenCarsDataset( dataroot='/floppy/datasets/PoseGen_resized/cars', 
                                  resize_dim=(256,256),
                                  shuffle=True,
                                  verbose=True)

    ## get object type ids
    object_type_ids = dataset.get_object_type_ids()
    # display samples
    for sample in dataset:
        cv2.namedWindow(sample['object_description'])
        ref_img_bgr = cv2.cvtColor(np.array(sample['ref_image']), cv2.COLOR_RGB2BGR)
        target_img_bgr = cv2.cvtColor(np.array(sample['target_image']), cv2.COLOR_RGB2BGR)
        sil_img_bgr = cv2.cvtColor(np.array(sample['sil_image']), cv2.COLOR_RGB2BGR)
        cv2.imshow(sample['object_description'], cv2.hconcat([ref_img_bgr, target_img_bgr, sil_img_bgr]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    