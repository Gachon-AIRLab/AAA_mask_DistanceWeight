import cv2
import numpy as np
import math
from GilAAADataset import *



def DTF(img):
    
    rows, cols = img.shape

    res= cv2.distanceTransform(img, cv2.DIST_L2, 5) #return 0.0~1.0 range value

    res = np.reciprocal(res)

    # res100= cv2.normalize(res, None, 100, 255, cv2.NORM_MINMAX, cv2.CV_8UC1, mask=img)
    # res150= cv2.normalize(res, None, 150, 255, cv2.NORM_MINMAX, cv2.CV_8UC1, mask=img)
    # res200= cv2.normalize(res, None, 200, 255, cv2.NORM_MINMAX, cv2.CV_8UC1, mask=img)
    res= cv2.normalize(res, None, 250, 255, cv2.NORM_MINMAX, cv2.CV_8UC1, mask=img)




    return res





if __name__ == "__main__":
    # mask = cv2.imread("/media/jihu/data/dataset/AAA/AAAGilDatasetPos/mask/05390853_20200821_0085.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread("/media/jihu/data/dataset/AAA/AAAGilDatasetPos/raw/05390853_20200821_0085.png")
    
    
    
    # o_mask = mask.copy()
    # o_img = img.copy()

    ## 제대로 안되넹?
    

    # cv2.imshow('or', o_mask)
    # cv2.imshow("te", res)
    # cv2.waitKey(0)
    
    dataset = GilAAADataset('/media/jihu/data/dataset/AAA/AAAGilDatasetPos')
    
  
    for img, target in dataset:
            # print(type(img)) => PIL
            # print(type(target["masks"])) => Tensor
            # print(target["labels"])
            
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imshow("or", img)
        path = target["path"]
        masksss = cv2.imread("/media/jihu/data/dataset/AAA/AAAGilDatasetPos/mask/" + target["path"], cv2.IMREAD_GRAYSCALE)

        masksss = DTF(masksss)
        
        # cv2.imshow("te230", masksss)
        # cv2.waitKey(0)
        #exit(0)
           
           
        cv2.imwrite("/media/jihu/data/dataset/AAA/AAAGilDatasetPos/WDmask250/" + target["path"], masksss)
           

        #exit(0)

