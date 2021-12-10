# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:11:16 2021

@author: Katarina
"""


import numpy as np
import matplotlib.pyplot as plt

import gzip


dirPathDistilled = "distilled_images/"
dirPathAncestor = "ancestor_images/"

"""
## Plots an image or set of images (all 28x28)
## input is either a single image, i.e., np.array with shape (28,28), or a square number of images, i.e., np.array with shape (z*z, 28, 28) for some integer z > 1
"""
def plot_image(im, fname='out.png', show=False):
    fig = plt.figure()
    im = im.reshape((-1,28, 28))

    num = im.shape[0]
    assert num <= 3 or np.sqrt(num)**2 == num, 'Number of images is too large or not a perfect square!'
    if num <= 3:
        for i in range(0, num):
            plt.subplot(1, num, 1 + i)
            plt.axis('off')
            plt.imshow(im[i], cmap='gray_r') # plot raw pixel data
    else:
        sq = int(np.sqrt(num))
        for i in range(0, num):
            plt.subplot(sq, sq, 1 + i)
            plt.axis('off')
            plt.imshow(im[i], cmap='gray_r') # plot raw pixel data

    #ensure_exists('./plots')
    # out_fp = './plots/{}'.format(fname)
    # plt.savefig(out_fp)

    if show is False:
        plt.close()

# plot 10 images (1 from each class) for the distilled images we are using: 10_distilled_batch_size_64_5epochs.gz
with gzip.open(dirPathDistilled + '10_distilled_batch_size_64_5epochs_1channel_labels.gz', 'rb') as lbpath:   # train
    train_labels = np.frombuffer(lbpath.read(), np.uint8)
    
with gzip.open(dirPathDistilled + '10_distilled_batch_size_64_5epochs_1channel_images.gz', 'rb') as imgpath:  # train
    train_imgs = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(train_labels), 28, 28)
print(train_imgs.shape)
plot_image(train_imgs[0:4])

# plot 10 images (1 from each class) for the ancestor images we are using: first_1000_train_images.gz