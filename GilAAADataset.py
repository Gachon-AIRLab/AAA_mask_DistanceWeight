import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


# from engine import train_one_epoch, evaluate
# import utils
import transforms as T

def get_transform(train):
    transforms = []
    
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        
        transforms.append(T.RandomHorizontalFlip(0.5))
        #transforms.append(T.RandomRotate(0.5))
        #transforms.append(T.RandomScaling(0.5))

    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())

    return T.Compose(transforms)



class GilAAADataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "raw"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.weightedmasks = list(sorted(os.listdir(os.path.join(root, "WDmask250"))))
#weightedmask

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "raw", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        weightedmask_path = os.path.join(self.root, "WDmask250", self.weightedmasks[idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB, 
        # because each color corresponds to a different instance
        # with 0 being background

        mask = Image.open(mask_path)
        weightedmask = Image.open(weightedmask_path)

        mask = np.array(mask) / 255
        mask = mask.astype(np.uint8)
        weightedmask = np.array(weightedmask)
        weightedmask = weightedmask.astype(np.uint8)


        # instances are encoded as different colors
        obj_ids = np.unique(mask)
  
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
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
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        weightedmask = torch.as_tensor(weightedmask, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = np.float32(0.0== obj_ids[:, None, None])
        if num_objs > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        weightedmask = weightedmask.unsqueeze(0)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = weightedmask #masks  
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
  

        #====using save image
        # path = img_path.split('/')
        # target["path"] = path[-1]


        #==== showing mask for check
        # print(target["masks"])
        # plt.imshow(target["masks"].numpy()[0])
        # plt.show()
        # exit(0)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)