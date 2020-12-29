import random
import torch

import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as tf

toPIL = tf.ToPILImage()
toTen = tf.ToTensor()

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


def returnBoxes(mask): #PIL image
    mask = np.array(mask)
    mask = mask.astype(np.uint8)
    # instances are encoded as different colors
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
    return boxes



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):

        if random.random() < self.prob:
            # oimage = image
            # otarget = target["masks"]
            # oimage.show()
            # plt.imshow(otarget.numpy()[0])
            # plt.show()
            
            flip = tf.RandomHorizontalFlip(1) #only horizen flip => always same both image and maak flip
            
            
            image = flip(image)
            mask = flip(toPIL(target["masks"]))
            target["masks"] = toTen(mask)
            target["boxes"] = returnBoxes(mask)
            
            # image.show()
            # plt.imshow(target["masks"].numpy()[0])
            # plt.show()
            #exit()

        return image, target


class RandomScaling(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):

        if random.random() < self.prob:
            # oimage = image
            # otarget = target["masks"]
            # oimage.show()
            # plt.imshow(otarget.numpy()[0])
            # plt.show()

            size = random.uniform(0.9, 1.1)
            size = int(512*size)                 

            scaling = tf.Resize(size)
            
            #scale
            image = scaling(image)
            mask = scaling(toPIL(target["masks"]))

            #set size
            image = F.center_crop(image, 512)
            mask = F.center_crop(mask, 512)


            target["boxes"] = returnBoxes(mask)
            target["masks"] = toTen(mask)
            boxes = target["boxes"]

            area = np.float32(0.0)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["area"] = area
            
            # image.show()
            # plt.imshow(target["masks"].numpy()[0])
            # plt.show()
            # exit(0)

        return image, target


class RandomRotate(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):

        if random.random() < self.prob:
            # oimage = image
            # otarget = target["masks"]
            # oimage.show()
            # plt.imshow(otarget.numpy()[0])
            # plt.show()

            RANDOM_SEED = random.random()

            rotate = tf.RandomRotation((-30, 30))
            
            #image rotate
            random.seed(RANDOM_SEED)            
            image = rotate(image)

            #mask rotate
            mask = toPIL(target["masks"])
            random.seed(RANDOM_SEED) 
            mask = rotate(mask)
            
            #seed => for image and mask hava same rotation


            target["masks"] = toTen(mask)
            target["boxes"] = returnBoxes(mask)         

            # image.show()
            # plt.imshow(target["masks"].numpy()[0])
            # plt.show()
            # exit(0)

        return image, target




class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
