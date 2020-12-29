import random
import torch

import imageio
from imageio import augmenters as iaa

from PIL import Image
import numpy as np

import torchvision
import utils
import pickle
import cv2
from torchvision.transforms import functional as F



def _flip_coco_person_keypoints(kps, width):
   flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
   flipped_data = kps[:, flip_inds]
   flipped_data[..., 0] = width - flipped_data[..., 0]
   # Maintain COCO convention that if visibility == 0, then x, y = 0
   inds = flipped_data[..., 2] == 0
   flipped_data[inds] = 0
   return flipped_data


def getitem():
    check_dir = '../AAAGilDatasetPos/'
    subject = '05390853_20200821'
    img_idx = 96

    img_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/raw/' + subject + '_%04d.png'%img_idx
    mask_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/mask/' + subject + '_%04d.png'%img_idx    
    

    img = Image.open(img_name).convert("RGB")
    mask = Image.open(mask_name)

    mask = np.array(mask)
    mask = np.array(mask) / 255

    mask = mask.astype(np.uint8)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]

    masks = mask == obj_ids[:, None, None] #ndarray

        # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.ones((num_objs,), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)

    image_id = torch.tensor([idx])
    area = np.float32(0.0)
    if num_objs > 0:
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["masks"] = masks
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd


    ######################################### rotate
    image = img

    if random.random() < 100:
        height, width = image.shape[-2:] #get image size

       
        seq = iaa.Sequential([
            iaa.Affine(rotate=(-45, 45))            
        ])
        
        images, target = seq.augment_images(images = image, keypoints = target["keypoints"], mask)
        
        bbox = target["boxes"]
        bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
        target["boxes"] = bbox
        if "masks" in target:
            target["masks"] = target["masks"].flip(-1)
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = _flip_coco_person_keypoints(keypoints, width)
            target["keypoints"] = keypoints
    
    
    return image, target






img, target = getitem()

print(type(img))
print(type(target))