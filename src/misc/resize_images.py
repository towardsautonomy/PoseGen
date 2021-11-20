import cv2
import os
import glob

# dimensions
resize_width = 256
resize_height = 256

# input and output paths
input_path = '/floppy/datasets/PoseGen/background/santa-clara-square-rooftop'
output_path = '/floppy/datasets/PoseGen_resized/background/santa-clara-square-rooftop'

# create output directory if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# list all files in the input directory
input_images = glob.glob(os.path.join(input_path, '*.JPEG'))

# go through all files in the input directory
for input_image in input_images:
    # read the image
    image = cv2.imread(input_image)

    # resize the image
    image = cv2.resize(image, (resize_width, resize_height))

    # save the image in png format
    output_image = os.path.join(output_path, os.path.basename(input_image).split('.')[0] + '.JPEG')
    cv2.imwrite(output_image, image)

    # print status
    print('Resized image: {0}'.format(output_image))