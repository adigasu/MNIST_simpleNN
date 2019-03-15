'''
Utility function for NN
'''

import numpy as np
import gzip


######## functions ########
def read_data(filename, n_img, img_sz):
  with gzip.open(filename) as data:
    data.read(16)
    buf = data.read(img_sz[0] * img_sz[1] * img_sz[2] * n_img)
    imgs = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    imgs = imgs.reshape(n_img, img_sz[0], img_sz[1], img_sz[2])
    return imgs

def read_label(filename, n_img):
  with gzip.open(filename) as data:
    data.read(8)
    buf = data.read(1 * n_img)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    labels = labels.reshape(n_img,1)
    return labels

def init_fc_weight(size):
    # return np.random.standard_normal(size=size) * 0.01
    return np.random.standard_normal(size=size) * np.sqrt(2./size[1])

