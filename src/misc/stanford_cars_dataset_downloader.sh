#!/bin/sh
# train set
wget http://ai.stanford.edu/~jkrause/car196/cars_train.tgz
# test set
wget http://ai.stanford.edu/~jkrause/car196/cars_test.tgz
# devkit
wget http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
# extract all
for tgz in *.tgz; do tar -xzvf $tgz; done