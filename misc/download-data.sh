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
pip install gdown
gdown https://drive.google.com/uc?id=1FMrdt5PdWZFlw2qED7OMPvJnqrRd-44k
unzip TeslaPoseGen.zip