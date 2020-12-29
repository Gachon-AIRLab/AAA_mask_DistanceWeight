import numpy as np
import cv2
from random import randint
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_distortion(image, alpha, sigma, seed=None):
    rows = image.shape[0]
    cols = image.shape[1]
    ch = image.shape[2]

    res = np.zeros((rows, cols, ch))

    if seed is None:
        seed = np.random.randint(777)
    else:
        seed = np.random.randint(seed)


    # random displacement vector
    # 모든 픽셀에...
    dx = np.random.uniform(-1, 1, (rows, cols))
    dy = np.random.uniform(-1, 1, (rows, cols))

    dx_gauss = cv2.GaussianBlur(dx, (7,7), sigma)
    dy_gauss = cv2.GaussianBlur(dy, (7,7), sigma)

    n = np.sqrt(dx_gauss**2 + dy_gauss**2)

    ndx = dx_gauss/n * alpha
    ndy = dy_gauss/n * alpha

    indy, indx = np.indices((rows,cols))

    ch = image.shape[2]

    map_x = map_x.reshape(rows, cols).astype(np.float32)
    map_y = map_y.reshape(rows, cols).astype(np.float32)

    # res = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)
    res = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return res




def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)



if __name__=="__main__":
    mask = cv2.imread("/media/jihu/data/dataset/AAA/AAAGilDatasetPos/mask/05390853_20200821_0085.png")
    img = cv2.imread("/media/jihu/data/dataset/AAA/AAAGilDatasetPos/raw/05390853_20200821_0085.png")

    o_mask = mask.copy()
    o_img = img.copy()

    print(mask.shape)
    #res = elastic_distortion(img,0.5 ,2)

    ## 제대로 안되넹?
    res2 = elastic_transform(img,5,100)

    cv2.imshow('or',o_img)
    #cv2.imshow("te",res)
    cv2.imshow("te2",res2)
    cv2.waitKey(0)