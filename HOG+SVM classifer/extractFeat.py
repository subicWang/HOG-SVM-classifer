#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/18 14:33
"""
# Import the functions to calculate feature descriptions
from skimage.feature import hog
import numpy as np
from sklearn.externals import joblib
# To read image file and save image feature descriptions
import os
import time
# import glob
import pickle as pk
from config import *
# import matplotlib.pyplot as plt


def unpickle(file):
    fo = open(file, 'rb')
    dict = pk.load(fo, encoding='bytes')
    fo.close()
    return dict

def getData(filePath):
    TrainData = []
    for childDir in os.listdir(filePath):
        if childDir != 'test_batch':
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            # train = np.reshape(data[str.encode('data')], (10000, 3, 32 * 32))
            # If your python version do not support to use this way to transport str to bytes.
            # Think another way and you can.
            train = np.reshape(data[b'data'], (10000, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            datalebels = zip(train, labels, fileNames)
            TrainData.extend(datalebels)
        else:
            f = os.path.join(filePath, childDir)
            data = unpickle(f)
            test = np.reshape(data[b'data'], (10000, 3, 32 * 32))
            labels = np.reshape(data[b'labels'], (10000, 1))
            fileNames = np.reshape(data[b'filenames'], (10000, 1))
            TestData = zip(test, labels, fileNames)
    return TrainData, TestData
def getFeat(TrainData, TestData):
    for data in TestData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image)/255.0
        fd = hog(gray, orientations, pixels_per_cell, cells_per_block)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = str(filename[0], encoding = "utf-8") .split('.')[0]+'.feat'
        fd_path = os.path.join('./data/features/test/', fd_name)
        joblib.dump(fd, fd_path)
    print("Test features are extracted and saved.")
    for data in TrainData:
        image = np.reshape(data[0].T, (32, 32, 3))
        gray = rgb2gray(image)/255.0
        fd = hog(gray, orientations, pixels_per_cell, cells_per_block)
        fd = np.concatenate((fd, data[1]))
        filename = list(data[2])
        fd_name = str(filename[0], encoding="utf-8") .split('.')[0]+'.feat'
        fd_path = os.path.join('./data/features/train/', fd_name)
        joblib.dump(fd, fd_path)
    print("Train features are extracted and saved.")
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray
if __name__ == '__main__':
    t0 = time.time()
    filePath = r'G:\DataLib\cifar-10-batches-py'
    TrainData, TestData = getData(filePath)
    getFeat(TrainData, TestData)
    t1 = time.time()
    print("Features are extracted and saved.")
    print('The cast of time is:%f'%(t1-t0))








