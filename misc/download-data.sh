#!/bin/sh
mkdir -p /data/
cd /data/

# download stanford cars
mkdir -p cars
cd cars
# train set
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
# test set
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
# devkit
wget http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
# extract all
for tgz in *.tgz; do tar -xzvf $tgz; done

# download tesla dataset
cd ..
wget -O PoseGen_resized.zip https://www.dropbox.com/s/22qd3ulhc9sl9u3/PoseGen_resized.zip?dl=1 -q --show-progress
unzip PoseGen_resized.zip
