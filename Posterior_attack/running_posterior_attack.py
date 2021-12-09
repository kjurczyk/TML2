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
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import tensorflow as tf


"""
## Extract 'sz' targets from in/out data
"""
def get_targets(x_in, y_in, x_out, y_out, sz=2000):

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
## Perform the posterior attack
## Inputs:
##  - x_targets, y_targets: records to attack
##  - query_target_model: function to query the target model [invoke as: query_target_model(x)]
##  - threshold: decision threshold

##  Output:
##  - in_or_out_pred: in/out prediction for each target
"""
def do_posterior_attack(x_targets, y_targets, query_target_model, threshold=0.9):

    ## TODO ##
    pv = query_target_model(x_targets)  #given target record - gives us the posterior
    in_or_out_pred = np.zeros((x_targets.shape[0],))    # put prediction of IN or OUT in here

    for i in range(0,len(pv)):
        largest_index = np.argmax(pv[i])        #figure out the largest probability
        # print("i: ", i, ", largest index: ", largest_index, "and the value is ", pv[i][largest_index])
        if pv[i][largest_index] > threshold:
            in_or_out_pred[i] = 1
            # print("I think ", i, " is IN because the ratio is ", prob_train[i], " and threshold = ", threshold)
        else:
            in_or_out_pred[i] = 0
            # print("I think ", i, " is OUT because the ratio is ", prob_train[i])

    #is that probability greater than the threshold?
        
    return in_or_out_pred




"""
## Compute attack performance metrics, i.e., accuracy and advantage (assumes baseline = 0.5)
## Note: also returns the full confusion matrix
"""
def attack_performance(in_or_out_test, in_or_out_pred):
    cm = metrics.confusion_matrix(in_or_out_test, in_or_out_pred)
    accuracy = np.trace(cm) / np.sum(cm.ravel())
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    advantage = tpr - fpr

    return accuracy, advantage, cm


"""
Get all the files 
"""
training_data_filename = 'first_1000_train' # change the filename

dirPathDistilled = "../distilled_images/"
dirPathAncestor = "../ancestor_images/"
files = [
      training_data_filename + '_labels.gz', training_data_filename +'_images.gz',
      'second_1000_attack_labels.gz', 'second_1000_attack_images.gz',
      'third_1000_out_labels.gz', 'third_1000_out_images.gz',
      'fourth_1000_val_labels.gz','fourth_1000_val_images.gz',
      'out_labels_20k.gz', 'out_images_20k.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
      ]

with gzip.open(dirPathAncestor + files[0], 'rb') as lbpath:   # train
    first_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)
    #print(first_1000_train_labels.shape)
    #print(first_1000_train_labels)

with gzip.open(dirPathAncestor + files[1], 'rb') as imgpath:  # train
    first_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(first_1000_train_labels), 28, 28).reshape(len(first_1000_train_labels), 784).reshape(len(first_1000_train_labels), 28, 28, 1)
    #print(first_1000_train_images.shape)
    #print(first_1000_train_images)
    
with gzip.open(dirPathAncestor + files[2], 'rb') as lbpath: # attack
    second_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(dirPathAncestor + files[3], 'rb') as imgpath: # attack
    second_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(second_1000_train_labels), 28, 28)

with gzip.open(dirPathAncestor + files[4], 'rb') as lbpath: # out
    third_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(dirPathAncestor + files[5], 'rb') as imgpath: # out
    third_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(third_1000_train_labels), 28, 28).reshape(len(third_1000_train_labels), 784).reshape(len(third_1000_train_labels), 28, 28, 1)
    
with gzip.open(dirPathAncestor + files[6], 'rb') as lbpath: # validation
    fourth_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(dirPathAncestor + files[7], 'rb') as imgpath: # validation
    fourth_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(fourth_1000_train_labels), 28, 28).reshape(len(fourth_1000_train_labels), 784).reshape(len(fourth_1000_train_labels), 28, 28, 1)
    
with gzip.open(dirPathAncestor + files[10], 'rb') as lbpath: # test
    t10k_labels_idx1_ubyte = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #print(t10k_labels_idx1_ubyte.shape)

with gzip.open(dirPathAncestor + files[11], 'rb') as imgpath: # test
    t10k_images_idx3_ubyte = np.frombuffer(
    imgpath.read(), np.uint8, offset=16).reshape(len(t10k_labels_idx1_ubyte), 28, 28).reshape(len(t10k_labels_idx1_ubyte), 784).reshape(len(t10k_labels_idx1_ubyte), 28, 28, 1)
    
    
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

# load the model from the saved file
#modelDir = "../training/"
model = load_model('first_1000_train_0.741')   # model = load_model(modelDir + 'first_1000_train_5_8')

if model is None:
    print('Model files do not exist. Train the model first!')
    
# load the data from the images and labels
x_train = first_1000_train_images #[:1000] # first 1000 images from .gz images file
y_train = first_1000_train_labels # first 1000 labels from .gz file

x_test = t10k_images_idx3_ubyte # all 10,000 images from test.gz?
y_test = t10k_labels_idx1_ubyte # all 10,000 labels from test.gz?

# x_attack_train = third_1000_train_images # some images from mnist training data
# y_attack_train = third_1000_train_labels

x_out = third_1000_train_images
y_out = third_1000_train_labels

# get the training size
target_train_size = x_train.shape[0]

# .gz has overall shape (2000, 28, 28) -- 2k images, each is 28x28 pixels
print('Loaded .gz data; shape: {} [y: {}], test shape: {} [y: {}]'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
# Let's flatten the images for easier processing (labels don't change)
flat_vector_size = 28 * 28
# flattens the images
x_train = x_train.reshape(x_train.shape[0], flat_vector_size)
x_test = x_test.reshape(x_test.shape[0], flat_vector_size)
# x_attack_train = x_attack_train.reshape(x_attack_train.shape[0], flat_vector_size)
x_out = x_out.reshape(x_out.shape[0], flat_vector_size)

# Put the labels in "one-hot" encoding using keras' to_categorical()
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# y_attack_train = keras.utils.to_categorical(y_attack_train, num_classes)
y_out = keras.utils.to_categorical(y_out, num_classes)

# extract targets (some in, some out)
x_targets, y_targets, in_or_out_targets = get_targets(x_train, y_train, x_out, y_out)



    

# evaluate loaded model on test data

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# model.compile(optimizer, loss)
print(f"x_test.shape: {x_test.shape}")
# tf.reshape(x_test, [28,28])
print(f"y_test.shape: {y_test.shape}")
score = model.evaluate(x_test, y_test, verbose=0)
print(model.metrics_names)
print(score) 



query_target_model = lambda x: model.predict(x)

threshold = []
in_or_out = []
for i in range(0,100):
    t = 0.01*i
    threshold.append(t)
    # print(f"current threshold: {t}")
    in_or_out_pred = do_posterior_attack(x_targets, y_targets, query_target_model, threshold=t)
    
    accuracy, advantage, _ = attack_performance(in_or_out_targets, in_or_out_pred)
    # print(f"accuracy: {accuracy}")
    in_or_out.append(accuracy)
    
# print(threshold)
# print(in_or_out)
    
in_or_out_pred = do_posterior_attack(x_targets, y_targets, query_target_model, threshold=0.8)
accuracy, advantage, _ = attack_performance(in_or_out_targets, in_or_out_pred)
print('Posterior accuracy, advantage: {:.1f}%, {:.2f}'.format(100.0*accuracy, advantage))


fig = plt.figure
plt.plot(threshold,in_or_out)
plt.xlabel("Threshold")
plt.ylabel("Posterior Accuracy")
plt.title("Posterior Attack Accuracy for Different Thresholds")
plt.show()