
# coding: utf-8

# In[1]:

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# # Mask R-CNN - Train on Jalbert Dataset
# 

# In[2]:

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
from tqdm import tqdm,trange
from config import Config
import utils
import model as modellib
from model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

## Dataset
import jalbert
config = jalbert.jalbertConfig()
dim = 512

data_folder = "/dds/work/workspace/data_ja/"
groundtruth_path='/dds/work/workspace/data_ja/RF-CAT1-v1.0.csv'

scales = [0.8, 1., 1.25]

dataset = jalbert.jalbertDataset()
dataset.load_jalbert(dim, data_folder,groundtruth_path, scales)#, force_new_dataset=True)
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

nb_img = dataset.num_images
split_train = slice(1,2*nb_img//3)
split_val = slice(2*nb_img//3,nb_img)


#nb_img = 77
#split_train = slice(1,nb_img//2)
#split_val = slice(nb_img//2,nb_img)

dataset_train = jalbert.jalbertDataset()
dataset_train.load_jalbert(dim, data_folder,groundtruth_path, scales, split=split_train)#, force_new_dataset=True)
dataset_train.prepare()

dataset_val = jalbert.jalbertDataset()
dataset_val.load_jalbert(dim, data_folder,groundtruth_path, scales, split=split_val)#, force_new_dataset=True)
dataset_val.prepare()

#dataset_train = jalbert.copydataset(dataset,split_train)
#dataset_val = jalbert.copydataset(dataset,split_val)

print("Train")
print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))

print("Val")
print("Image Count: {}".format(len(dataset_val.image_ids)))
print("Class Count: {}".format(dataset_val.num_classes))
for i, info in enumerate(dataset_val.class_info):
    print("{:3}. {:50}".format(i, info['name']))


#config.display()

## Detection
class InferenceConfig(jalbert.jalbertConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")

model_path = model.find_last()[1]


model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180311T2313/mask_rcnn_jalbert_0025.h5" #p 0.9 r 0.4
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180319T2339/mask_rcnn_jalbert_0020.h5" # v2 p50 r54
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180319T2339/mask_rcnn_jalbert_0003.h5" # v2 p45 r59
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180319T2339/mask_rcnn_jalbert_0010.h5" # v2 p45 r59
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180320T1743/mask_rcnn_jalbert_0110.h5" #v2 p53 r63
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180320T1743/mask_rcnn_jalbert_0050.h5" #v2 p48 r61
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180320T1743/mask_rcnn_jalbert_0100.h5" #v2 p58 r70
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180320T1743/mask_rcnn_jalbert_0090.h5" #v2 p55 r63
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180320T1743/mask_rcnn_jalbert_0085.h5" #v2 p55 r62
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180320T1743/mask_rcnn_jalbert_0080.h5" #v2 p61 r73 #####
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180320T1743/mask_rcnn_jalbert_0075.h5" #v2 p58 r65
model_path = "/dds/work/workspace/Mask_RCNN/logs/jalbert20180320T1743/mask_rcnn_jalbert_0070.h5" #v2 p59 r68






#model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes_jalbert_all.h5")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


#dataset_val = dataset_train

nb_test_images = len(dataset_val.image_ids)
print("testing on "+str(nb_test_images)+" images...")
image_ids = np.random.choice(dataset_val.image_ids, nb_test_images)
APs = []
match_counts = []
pred_counts = []
gt_boxes_counts = []

e = 0.0000000000000000001
t = tqdm(image_ids)
for image_id in t: #qdm(image_ids):
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    
    match_count, pred_count, gt_boxes_count = utils.jg_compute_scores(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
    match_counts.append(match_count)
    pred_counts.append(pred_count)
    gt_boxes_counts.append(gt_boxes_count)
    t.set_postfix( mAP=np.mean(APs),match=np.sum(match_counts),pred=np.sum(pred_counts),gt=np.sum(gt_boxes_counts),p=(e+np.sum(match_counts))/(e+np.sum(pred_counts)),r=(e+np.sum(match_counts))/(e+np.sum(gt_boxes_counts)))




print("mAP: ", np.mean(APs))
print("match_count: ", np.sum(match_counts))
print("pred_count: ", np.sum(pred_counts))
print("gt_boxes_count: ", np.sum(gt_boxes_counts))
print("precision: ", np.sum(match_counts)/np.sum(pred_counts))
print("recall: ", np.sum(match_counts)/np.sum(gt_boxes_counts))

print("Weights used : ", model_path)


print("Over.")


