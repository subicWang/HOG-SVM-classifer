#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/19 11:08
"""
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np
import glob
import os
import time
from config import *

if __name__ == "__main__":
    t0 = time.time()
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    num = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        fds.append(data[:-1])
        labels.append(data[-1])
    if clf_type is 'LIN_SVM':
        clf = LinearSVC()
        print("Training a Linear SVM Classifier.")
        clf.fit(fds, labels)
        # If feature directories don't exist, create them
        # if not os.path.isdir(os.path.split(model_path)[0]):
        #     os.makedirs(os.path.split(model_path)[0])
        # joblib.dump(clf, model_path)
        # clf = joblib.load(model_path)
        print("Classifier saved to {}".format(model_path))
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            data_test = joblib.load(feat_path)
            temp = data_test[:-1]
            data_test_feat = temp.reshape((1, -1))
            result = clf.predict(data_test_feat)
            if int(result) == int(data_test[-1]):
                num += 1
        rate = float(num)/total
        t1 = time.time()
        print('The classification accuracy is %f'%rate)
        print('The cast of time is :%f'%(t1-t0))




