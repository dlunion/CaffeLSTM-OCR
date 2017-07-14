#coding=gbk
import sys
sys.path.insert(0, r"../../python")
import caffe
import time
import numpy as np
import cv2

modelFile = "deploy.prototxt"
pretrained = "models/_iter_22006.caffemodel"
reload(sys)
sys.setdefaultencoding("utf-8")

caffe.set_mode_gpu();

with open("label-map.txt", 'r') as f:
    label = f.read().split("\n")

net = caffe.Classifier(modelFile, pretrained, channel_swap=(2, 1, 0), raw_scale=256)
input_image = caffe.io.load_image("samples/Y43344.20_d41d8cd98f00b204e9800998ecf8427e.png")
input_image = input_image
r = net.predict([input_image], False)
out = [r[0][i, :].argmax() for i in range(r[0].shape[0])]
pout = ""
conf = []

prev = 15
for i in range(len(out)):
    if out[i] != prev and out[i] != 15:
        pout = pout + label[out[i]]
        conf.append(r[0][i][out[i]])
    prev = out[i]
print pout, conf