# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:52:21 2021

@author: Katarina
"""

"""
Opening a .h5 and loading from it
"""

import os
import keras
import gzip
import numpy as np


"""
## Extract 'sz' targets from in/out data
"""
def get_targets(x_in, y_in, x_out, y_out, sz=500):

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


"""
Get all the files 
"""
files = [
      'first_1000_train_labels.gz', 'first_1000_train_images.gz',
      'second_1000_train_labels.gz', 'second_1000_train_images.gz',
      'third_1000_train_labels.gz', 'third_1000_train_images.gz',
      'fourth_1000_train_labels.gz', 'fourth_1000_train_images.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
      ]

with gzip.open(files[0], 'rb') as lbpath:
    first_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)
    #print(first_1000_train_labels.shape)
    #print(first_1000_train_labels)

with gzip.open(files[1], 'rb') as imgpath:
    first_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(first_1000_train_labels), 28, 28)
    #print(first_1000_train_images.shape)
    #print(first_1000_train_images)
    
with gzip.open(files[2], 'rb') as lbpath:
    second_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(files[3], 'rb') as imgpath:
    second_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(second_1000_train_labels), 28, 28)

with gzip.open(files[4], 'rb') as lbpath:
    third_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(files[5], 'rb') as imgpath:
    third_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(third_1000_train_labels), 28, 28)
    
with gzip.open(files[6], 'rb') as lbpath:
    fourth_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(files[7], 'rb') as imgpath:
    fourth_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(fourth_1000_train_labels), 28, 28)
    
with gzip.open(files[8], 'rb') as lbpath:
    t10k_labels_idx1_ubyte = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #print(t10k_labels_idx1_ubyte.shape)

with gzip.open(files[9], 'rb') as imgpath:
    t10k_images_idx3_ubyte = np.frombuffer(
    imgpath.read(), np.uint8, offset=16).reshape(len(t10k_labels_idx1_ubyte), 28, 28)
    

x_train = first_1000_train_images #[:1000] # first 1000 images from .gz images file
y_train = first_1000_train_labels # first 1000 labels from .gz file

x_test = t10k_images_idx3_ubyte # all 10,000 images from test.gz?
y_test = t10k_labels_idx1_ubyte # all 10,000 labels from test.gz?

x_attack_train = third_1000_train_images # some images from mnist training data
y_attack_train = third_1000_train_labels

x_out = fourth_1000_train_images
y_out = fourth_1000_train_labels

target_train_size = x_train.shape[0]

# .gz has overall shape (2000, 28, 28) -- 2k images, each is 28x28 pixels
print('Loaded .gz data; shape: {} [y: {}], test shape: {} [y: {}]'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
# Let's flatten the images for easier processing (labels don't change)
flat_vector_size = 28 * 28
x_train = x_train.reshape(x_train.shape[0], flat_vector_size)
x_test = x_test.reshape(x_test.shape[0], flat_vector_size)
x_attack_train = x_attack_train.reshape(x_attack_train.shape[0], flat_vector_size)
x_out = x_out.reshape(x_out.shape[0], flat_vector_size)

# Put the labels in "one-hot" encoding using keras' to_categorical()
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_attack_train = keras.utils.to_categorical(y_attack_train, num_classes)
y_out = keras.utils.to_categorical(y_out, num_classes)

# extract targets (some in, some out)
x_targets, y_targets, in_or_out_targets = get_targets(x_train, y_train, x_out, y_out)


"""
## Load model from file
"""        
def load_model(base_fp):
    # Model reconstruction from JSON file
    arch_json_fp = '{}-architecture.json'.format(base_fp)
    if not os.path.isfile(arch_json_fp):
        return None
        
    with open(arch_json_fp, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # Load weights into the new model
    model.load_weights('{}-weights.h5'.format(base_fp))
    
    print('Loaded model from file ({}).'.format(base_fp))
    return model

model = load_model('30e_025rc_300h')
if model is None:
    print('Model files do not exist. Train the model first!')
    

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
metrics=['accuracy'])

score = model.evaluate(x_test, y_test, verbose=0)
print(score) 
print(model.metrics_names)
