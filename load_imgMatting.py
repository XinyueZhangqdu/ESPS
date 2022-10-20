import numpy as np

def load():
    y_mask_train = np.load('dataset/data/npy/train_mask.npy')
    x_img_train = np.load('dataset/data/npy/train_img.npy')
    edge_train = np.load('dataset/data/npy/train_edge.npy')
    return y_mask_train,  x_img_train, edge_train
