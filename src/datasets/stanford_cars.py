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
class StanfordCarsDataset(Dataset):

    def __init__(self, obj_dataroot, 
                       bgnd_dataroot,
                       sil_dataroot,
                       resize_dim, 
                       transforms=None,
                       object_id=None, 
                       shuffle=True,
                       background_ext='jpg',
                       silhouette_ext='jpg',
                       verbose=True):
        """
        Args:
            obj_dataroot (string): Root Directory of Stanford Cars dataset.
            bgnd_dataroot (string): Root Directory of background image dataset.
            sil_dataroot (string): Root Directory of silhouette image dataset.
            resize_dim (tuple(w, h)): Dimension to resize the images to.
            object_id (int): Object ID to retrieve data for - set to None if all classes are needed.
            shuffle (bool): Whether or not to shuffle the dataset.
            verbose (bool): Whether or not to print additional information.
        """
        self.obj_dataroot = obj_dataroot
        self.bgnd_dataroot = bgnd_dataroot
        self.sil_dataroot = sil_dataroot
        self.resize_dim = resize_dim
        self.transforms = transforms
        self.object_id = object_id
        self.shuffle = shuffle
        self.verbose = verbose

        metadata_filename = os.path.join(obj_dataroot, 'car_devkit/devkit/cars_meta.mat')
        annotations_filename = os.path.join(obj_dataroot, 'car_devkit/devkit/cars_train_annos.mat')
        train_dir = os.path.join(obj_dataroot, 'cars_train')
        test_dir = os.path.join(obj_dataroot, 'cars_test')

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

        # get background images
        self.background_image_filenames = glob.glob(os.path.join(bgnd_dataroot, '*.'+background_ext))

        # get silhouette images
        self.silhouette_image_filenames = glob.glob(os.path.join(sil_dataroot, '*.'+silhouette_ext))

        # shuffle
        if shuffle == True:
            np.random.shuffle(self.id_filaneme_pairs)
            np.random.shuffle(self.background_image_filenames)
            np.random.shuffle(self.silhouette_image_filenames)

        # print stats
        if self.verbose:
            print('Number of object samples found in the dataset: {}'.format(self.__len__()))
            print('Number of background samples found in the dataset: {}'.format(len(self.background_image_filenames)))
            print('Number of silhouette samples found in the dataset: {}'.format(len(self.silhouette_image_filenames)))

    # method to get length of data
    def __len__(self, object_id=None):
        if object_id==None:
            return len(self.id_filaneme_pairs)
        else:
            assert(object_id in self.car_type_ids)
            return len(self.annotations[object_id])

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

    # method to get a list of background image filenames
    def get_background_filenames(self):
        return self.background_image_filenames

    # method to get a list of silhouette image filenames
    def get_silhouette_filenames(self):
        return self.silhouette_image_filenames


    # method to get a dictionary of {object type: object_description} pairs
    def object_id_description_dict(self):
        return self.id_description_dict

# main function
if __name__ == '__main__':
    import cv2

    ## dataset object
    dataset = StanfordCarsDataset( obj_dataroot='/floppy/datasets/Stanford', 
                                   bgnd_dataroot='/floppy/datasets/PoseGen/background',
                                   sil_dataroot='/floppy/datasets/PoseGen/rendered_silhouette',
                                   resize_dim=(256,256),
                                   verbose=True)

    ## get object type ids
    object_type_ids = dataset.get_object_type_ids()
    # display samples
    for i in range(dataset.__len__(1)):
        sample = dataset.__getitem__(i, object_type_ids[0])
        cv2.namedWindow(sample['object_description'])
        obj_img_bgr = cv2.cvtColor(np.array(sample['obj_image']), cv2.COLOR_RGB2BGR)
        bgnd_img_bgr = cv2.cvtColor(np.array(sample['bgnd_image']), cv2.COLOR_RGB2BGR)
        sil_img_bgr = cv2.cvtColor(np.array(sample['sil_image']), cv2.COLOR_RGB2BGR)
        cv2.imshow(sample['object_description'], cv2.hconcat([obj_img_bgr, bgnd_img_bgr, sil_img_bgr]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    