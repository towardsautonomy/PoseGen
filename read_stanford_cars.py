import os
import scipy.io

# configure paths
dataroot = '/floppy/datasets/Stanford'
metadata_filename = os.path.join(dataroot, 'car_devkit/devkit/cars_meta.mat')
annotations_filename = os.path.join(dataroot, 'car_devkit/devkit/cars_train_annos.mat')
train_dir = os.path.join(dataroot, 'cars_train')
test_dir = os.path.join(dataroot, 'cars_test')

# read metadata
cars_metadata = {}
mat = scipy.io.loadmat(metadata_filename)
for i, car_model in enumerate(mat['class_names'][0]):
    cars_metadata[i+1] = car_model[0]

# read annotations
annotations = {}
mat = scipy.io.loadmat(annotations_filename)
for ann in mat['annotations'][0]:
    filename = os.path.join(train_dir, ann[-1][0])
    car_type_id = ann[-2][0][0]
    if car_type_id in annotations.keys():
        annotations[car_type_id].append(filename)
    else:
        annotations[car_type_id] = [filename]

print(cars_metadata[10])
for ann in annotations[10]: print(ann)
    