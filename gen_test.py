# import the necessary packages
import tensorflow as tf
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
import numpy as np
import colorsys
import argparse
import imutils
import random
#import cv2
import os

class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_inference"
    
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # number of classes (we would normally add +1 for the background
    # but the background class is *already* included in the class
    # names)
    NUM_CLASSES = 1+80

config = myMaskRCNNConfig()

print("loading weights for Mask R-CNN model…")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="./")
model.load_weights('./mask_rcnn_coco.h5', by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
 'bus', 'train', 'truck', 'boat', 'traffic light',
 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
 'kite', 'baseball bat', 'baseball glove', 'skateboard',
 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
 'teddy bear', 'hair drier', 'toothbrush']

from skimage import transform
import os
import imageio

count = 0
source = "/home/ec2-user/Mask_RCNN/test"
for i in range(10):
    
    print('now we are in the folder C',i)
    imgs = os.listdir(source)
    print(f'number of imgs = {len(imgs)}, first image ={imgs[0]}')
    for j in range(len(imgs)):
    #for j in range(1000):

        img_name = source+"/"+imgs[j]
        #print(f'img={img_name}')
        im = imageio.imread(img_name)
        # run prediction
        results = model.detect([im], verbose=0)
        r = results[0]
        # get the masks
        #mask = r['masks']
        #mask = mask.astype(int)

        # find the mask corresponding to person
        classes= r['class_ids']
        idx = 0
        y1 = 0
        y2 = 0
        x1 = 0
        x2 = 0
        found_person = False
        for k in range(len(classes)):
            if (class_names[classes[k]] == 'person'):
                idx = k
                found_person = True
                y1, x1, y2, x2 = r['rois'][idx]
                break

        dest_path = '/home/ec2-user/Mask_RCNN/mask/zoom/test'+"/"+imgs[j]
        temp = imageio.imread(img_name)
        if (found_person):

            

            #for l in range(temp.shape[2]):
            #    temp[:,:,l] = temp[:,:,l] * mask[:,:,idx]

            temp = temp[y1:y2,x1:x2]
            resize = transform.resize(temp, (224,224), mode='symmetric', preserve_range=True)
            resize = resize.astype(np.uint8)
             
            #print(f'dest path = {dest_path}')
            imageio.imwrite(dest_path, resize)
        else:
            # just resize the old image
            temp = temp[50:,120:-50]
            resize = transform.resize(im, (224,224), mode='symmetric', preserve_range=True)
            resize = resize.astype(np.uint8)
             
            #print(f'dest path = {dest_path}')
            imageio.imwrite(dest_path, resize)
        #print(f'count = {count}')
        count += 1
        #if ((count % 100) == 0):
        print(f'count = {count}')
        


          