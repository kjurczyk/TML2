# -*- coding: utf-8 -*-
""" CIS6930TML -- Homework 2 -- hw2.py

# This file is the main homework file
"""


# This configuration works to train a model: problem1 simple,760,0.001 1000 50
# this also works for problem3: problem3 simple,32,0.0 1000 50


import os
import sys

import numpy as np
import matplotlib.pyplot as plt
#library for creating tables
from tabulate import tabulate
from PIL import Image
import glob
import tensorflow as tf

# we'll use keras for neural networks
import keras
from keras.datasets import mnist
from keras.datasets import fashion_mnist
# example of converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image


# our neural network architectures
import nets
import attacks

## os / paths
def ensure_exists(dir_fp):
    if not os.path.exists(dir_fp):
        os.makedirs(dir_fp)

## parsing / string conversion to int / float
def is_int(s):
    try:
        z = int(s)
        return z
    except ValueError:
        return None


def is_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return None


"""
## Load and preprocess the dataset
"""
def load_preprocess_mnist_data(train_in_out_size=2000):

    # load the image
    path = '/blue/cis6940/heting.wang/results/distill_basic/70k/'
    dataset = tf.keras.preprocessing.image_dataset_from_directory(path,label_mode="categorical",batch_size=70000,shuffle=False,class_names=["class_4","class_2","class_3","class_0","class_1","class_5","class_7","class_6","class_8","class_9"],color_mode="grayscale",image_size=(28, 28))
    for images, labels in dataset.take(1):  # only take first element of dataset
        numpy_images = images.numpy()
        numpy_labels = labels
      
    # this is the line that gets changed to mnist.load_data() if you want to load the mnist data instead of the fashion_mnist data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
    # x_train = numpy_images[:60000,:]
    # x_test = numpy_images[60000:,:]
    # y_train = numpy_labels[:60000,:]
    # y_test = numpy_labels[60000:,:]
    
    # # Let's flatten the images for easier processing (labels don't change)
    flat_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], flat_vector_size)
    x_test = x_test.reshape(x_test.shape[0], flat_vector_size)
    
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    # MNIST has overall shape (60000, 28, 28) -- 60k images, each is 28x28 pixels
    print('Loaded mnist data; train shape: {}y[{}], test shape: {}y[{}]'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    # Put the labels in "one-hot" encoding using keras' to_categorical()
    num_classes = 10

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    # let's split the training set further
    aux_idx = train_in_out_size

    x_aux = x_train[aux_idx:,:]
    y_aux = y_train[aux_idx:,:]

    x_temp = x_train[:aux_idx,:]
    y_temp = y_train[:aux_idx,:]

    out_idx = int(aux_idx/2.0)
    x_out = x_temp[out_idx:,:]
    y_out = y_temp[out_idx:,:]

    x_train = x_temp[:out_idx,:]
    y_train = y_temp[:out_idx,:]

    return (x_train, y_train), (x_out, y_out), (x_test, y_test), (x_aux, y_aux)


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

    ensure_exists('./plots')
    out_fp = './plots/{}'.format(fname)
    plt.savefig(out_fp)

    if show is False:
        plt.close()



"""
## Extract 'sz' targets from in/out data
"""
def get_targets(x_in, y_in, x_out, y_out, sz=5000):

    x_temp = np.vstack((x_in, x_out))
    y_temp = np.vstack((y_in, y_out))

    inv = np.ones((x_in.shape[0],1))
    outv = np.zeros((x_out.shape[0],1))
    in_out_temp = np.vstack((inv, outv))

    assert x_temp.shape[0] == y_temp.shape[0]

    if sz > x_temp.shape[0]:
        sz = x_temp.shape[0]

    perm = np.random.permutation(x_temp.shape[0])
    perm = perm[0:sz]
    x_targets = x_temp[perm,:]
    y_targets = y_temp[perm,:]

    in_out_targets = in_out_temp[perm,:]

    return x_targets, y_targets, in_out_targets


## this is the main
def main():

    # figure out the problem number
    assert len(sys.argv) >= 5, 'Incorrect number of arguments!'
    p_split = sys.argv[1].split('problem')
    assert len(p_split) == 2 and p_split[0] == '', 'Invalid argument {}.'.format(sys.argv[1])
    problem_str = p_split[1]

    assert is_number(problem_str) is not None, 'Invalid argument {}.'.format(sys.argv[1])
    problem = float(problem_str)
    probno = int(problem)

    if probno <= 0 or probno > 4:
        assert False, 'Problem {} is not a valid problem # for this assignment/homework!'.format(problem)

    ## change this line to override the verbosity behavior
    verb = True if probno == 1 else False

    # get arguments
    target_model_str = sys.argv[2]
    if target_model_str.startswith('simple'):
        simple_args = target_model_str.split(',')
        assert simple_args[0] == 'simple' and len(simple_args) == 3, '{} is not a valid network description string!'.format(target_model_str)
        hidden = is_int(simple_args[1])
        reg_const = is_number(simple_args[2])
        assert hidden is not None and hidden > 0 and reg_const is not None and reg_const >= 0.0, '{} is not a valid network description string!'.format(target_model_str)
        target_model_train_fn = lambda: nets.get_simple_classifier(num_hidden=hidden, l2_regularization_constant=reg_const,
                                                                   verbose=verb)
    elif target_model_str == 'deep':
        target_model_train_fn = lambda: nets.get_deeper_classifier(verbose=verb)
    else:
        assert False, '{} is not a valid network description string!'.format(target_model_str)

    target_model = target_model_train_fn() # compile the target model

    train_in_out_size = is_int(sys.argv[3])
    num_epochs = is_int(sys.argv[4])

    assert train_in_out_size is not None and 100 <= train_in_out_size <= 10000, '{} is not a valid size for the target model training dataset!'.format(sys.argv[3])
    assert num_epochs is not None and 0 < num_epochs <= 10000, '{} is not a valid size for the number of epochs to train the target model!'.format(sys.argv[4])

    # load the dataset
    train, out, test, aux = load_preprocess_mnist_data(train_in_out_size=2*train_in_out_size)
    x_train, y_train = train
    target_train_size = x_train.shape[0]
    x_out, y_out = out
    x_test, y_test = test
    x_aux, y_aux = aux
    
    
    
    # extract targets (some in, some out)
    x_targets, y_targets, in_or_out_targets = get_targets(x_train, y_train, x_out, y_out)

    assert train_in_out_size == target_train_size, 'Inconsistent training data size!'

    # train the target model
    train_loss, train_accuracy, test_loss, test_accuracy = nets.train_model(target_model, x_train, y_train, x_test, y_test, num_epochs, verbose=verb)

    print('Trained target model on {} records. Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size,
                                                                                    100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))
    
    query_target_model = lambda x: target_model.predict(x)

    if probno == 3:  ## only perform posterier attack for now so I removed other attacks

        assert len(sys.argv) == 5, 'Invalid extra argument'

        # TODO ##
        #compute the best threshold (for posterior_attack)
        max_accuracy = 0
        max_advantage = 0
        tlist = np.linspace(0,1,11)
        for threshold in tlist:
            in_or_out_pred = attacks.do_posterior_attack(x_targets, y_targets, query_target_model, threshold)
            if attacks.attack_performance(in_or_out_targets, in_or_out_pred)[0] > max_accuracy:
                max_accuracy, max_advantage, _ = attacks.attack_performance(in_or_out_targets, in_or_out_pred)
    
                print('posterior attack accuracy, advantage: {:.1f}%, {:.2f}, {:.2f}'.format(100.0*max_accuracy, max_advantage, threshold))

    
if __name__ == '__main__':
    main()
