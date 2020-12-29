
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from GilAAADataset import *
from engine import train_one_epoch, evaluate
import utils
import pickle
import cv2
import transforms as T
from torchvision.transforms import functional as F
import torchvision.transforms as tf


from gil_eval import *
from PIL import Image


dataset = GilAAADataset('/media/jihu/data/dataset/AAA/AAAGilDatasetPos/mix', get_transform(train=False))

i = 0

for img, target in dataset:

    img = cv2.imread("/media/jihu/data/dataset/AAA/AAAGilDatasetPos/mix/raw/" + target["path"], cv2.IMREAD_COLOR)
    mask = cv2.imread("/media/jihu/data/dataset/AAA/AAAGilDatasetPos/mix/mask/" + target["path"], cv2.IMREAD_GRAYSCALE)

    cv2.imshow("raw", img)
    cv2.imshow("mask", mask)
    
    if i==0:
        cv2.waitKey(0)
    i = 2;    
        
    cv2.waitKey(50)
    
    
