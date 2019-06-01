#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/18 21:03
Set the config variable.
"""
import configparser as cp
import json

config = cp.RawConfigParser()
config.read('./data/config/config.cfg')

orientations = json.loads(config.get("hog", "orientations"))
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
normalize = config.getboolean("hog", "normalize")
train_feat_path = config.get("path", "train_feat_path")
test_feat_path = config.get("path", "test_feat_path")
model_path = config.get("path", "model_path")
