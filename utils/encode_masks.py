import torch
import torch.nn.functional as fun
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def encode_masks(img, colours, num_classes = 22):

    label = np.zeros_like(img)
    print(label.shape)
    for idx, color in enumerate(colours):

        label[np.sum(img == np.array([[color]]), 2) == 3] = idx
    print(label.shape)
    onehot = np.eye(num_classes)[label]
    return onehot


def one_hot(img, colours):

    h, w = img.shape[:2]
    x = img.copy()

    
    x[np.where()]
    
    print(x.shape)
    return x

colours = np.array([[120, 120, 120], [127, 0, 0], [254, 0, 0], [0, 84, 0], [169, 0, 50], [254, 84, 0], [255, 0, 84], [0, 118, 220], [84, 84, 0], [0, 84, 84], [84, 50, 0], [51, 85, 127], [0, 127, 0], [0, 0, 254], [50, 169, 220], [0, 254, 254], [84, 254, 169], [169, 254, 84], [254, 254, 0], [254, 169, 0], [102, 254, 0], [182, 255, 0]])
print(colours[0])
img = cv2.imread('/home/vrushank/Downloads/instance-level_human_parsing/Training/Categories/0000010.png')

label = one_hot(img, colours)
print(label.shape)
plt.imshow(label, cmap = 'gray')
plt.show()
