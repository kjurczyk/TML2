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
from PIL import Image

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

        if pv[i][largest_index] >= threshold:
            in_or_out_pred[i] = 1

        else:
            in_or_out_pred[i] = 0

        
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
files = [
      'first_1000_train_labels.gz', 'first_1000_train_images.gz',
      'second_1000_train_labels.gz', 'second_1000_train_images.gz',
      'third_1000_train_labels.gz', 'third_1000_train_images.gz',
      'out_labels_20k.gz', 'out_images_20k.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
      ]
with gzip.open(files[0], 'rb') as lbpath:
     first_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)
with gzip.open(files[1], 'rb') as imgpath:
    first_1000_train_images = np.frombuffer(imgpath.read(), np.uint8).reshape(len(first_1000_train_labels), 28, 28)
   
# with gzip.open(files[2], 'rb') as lbpath:
#     second_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

# with gzip.open(files[3], 'rb') as imgpath:
#     second_1000_train_images = np.frombuffer(
#     imgpath.read(), np.uint8).reshape(len(second_1000_train_labels), 28, 28)

# with gzip.open(files[4], 'rb') as lbpath:
#     third_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

# with gzip.open(files[5], 'rb') as imgpath:
#     third_1000_train_images = np.frombuffer(
#     imgpath.read(), np.uint8).reshape(len(third_1000_train_labels), 28, 28)
    
with gzip.open(files[4], 'rb') as lbpath:
    third_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(files[5], 'rb') as imgpath:
    third_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(third_1000_train_labels), 28, 28)
    
with gzip.open(files[8], 'rb') as lbpath:
    t10k_labels_idx1_ubyte = np.frombuffer(lbpath.read(), np.uint8, offset=8)

with gzip.open(files[9], 'rb') as imgpath:
    t10k_images_idx3_ubyte = np.frombuffer(
    imgpath.read(), np.uint8, offset=16).reshape(len(t10k_labels_idx1_ubyte), 28, 28)
    
    
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
model = load_model('10_distilled')
modelo = load_model('first_1000')
if model is None or modelo is None:
    print('Model files do not exist. Train the model first!')
    
# load the data from the images and labels

path = os.getcwd()
dataset = tf.keras.preprocessing.image_dataset_from_directory(path,labels='inferred',label_mode="categorical",batch_size=10,shuffle=False,class_names=['0','1','2','3','4','5','6','7','8','9'],color_mode="grayscale",image_size=(28, 28))
for images, labels in dataset.take(1):  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels
      
path1 = "D:\out"
dataset1 = tf.keras.preprocessing.image_dataset_from_directory(path1,labels='inferred',label_mode="categorical",batch_size=10,shuffle=False,class_names=['0','1','2','3','4','5','6','7','8','9'],color_mode="grayscale",image_size=(28, 28))
for images1, labels1 in dataset1.take(1):  # only take first element of dataset
    numpy_images1 = images1.numpy()
    numpy_labels1 = labels1
   
x_train = first_1000_train_images
y_train = first_1000_train_labels

x_test = t10k_images_idx3_ubyte # all 10,000 images from test.gz
y_test = t10k_labels_idx1_ubyte # all 10,000 labels from test.gz



x_out = third_1000_train_images
y_out = third_1000_train_labels

# get the training size
target_train_size = x_train.shape[0]

# train set has overall shape (1000, 28, 28) -- 1k images, each is 28x28 pixels
print('Loaded .gz data; shape: {} [y: {}], out shape: {} [y: {}]'.format(x_train.shape, y_train.shape, x_out.shape, y_out.shape))
# Let's flatten the images for easier processing (labels don't change)
flat_vector_size = 28 * 28
# flattens the images
x_train = x_train.reshape(x_train.shape[0], flat_vector_size)
x_test = x_test.reshape(x_test.shape[0], flat_vector_size)
x_out = x_out.reshape(x_out.shape[0], flat_vector_size)

# Create just a figure and only one subplot
x1 = x_train[:1,:]
x2 = x_train[100:101,:]
x3 = x_train[200:201,:]
x4 = x_train[300:301,:]
x5 = x_train[400:401,:]
x6 = x_train[500:501,:]
x7 = x_train[600:601,:]
x8 = x_train[700:701,:]
x9 = x_train[800:801,:]
x10 = x_train[900:901,:]


fig, axs = plt.subplots(2,5)

images = np.vstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10))
    
for j in range(2):
    for z in range(5):
        data = Image.fromarray(np.uint8(images[j*5+z,:]).reshape(28,28))
        axs[j,z].imshow(data)
        axs[j,z].axis('off')











# Put the labels in "one-hot" encoding using keras' to_categorical()
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_out = keras.utils.to_categorical(y_out, num_classes)

# extract targets (some in, some out)
x_targets, y_targets, in_or_out_targets = get_targets(x_train, y_train, x_out, y_out)



    

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

score = model.evaluate(x_test, y_test, verbose=0)
print("distilled model.metrics  ", model.metrics_names)
print("score  ", score) 

modelo.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

score = modelo.evaluate(x_test, y_test, verbose=0)
print("original model.metrics  ", modelo.metrics_names)
print("score  ", score) 

query_target_model = lambda x: model.predict(x)
query_target_modelo = lambda x: modelo.predict(x)





threshold = []
in_or_out = []
#compute the best threshold (for posterior_attack)
max_accuracy = 0
max_advantage = 0
best_threshold = 0
tlist = np.linspace(0,1,100)

fig = plt.figure
plt.title("Distilled model")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
for threshold in tlist:
    in_or_out_pred = do_posterior_attack(x_targets, y_targets, query_target_model, threshold)
    accuracy = attack_performance(in_or_out_targets, in_or_out_pred)[0]
    
    plt.plot(threshold,accuracy,marker = '+', linestyle = '-')
    
    if accuracy > max_accuracy and threshold !=0:
        max_accuracy, max_advantage, _ = attack_performance(in_or_out_targets, in_or_out_pred)
        best_threshold = threshold
print('distilled posterior attack accuracy, advantage, best threshold: {:.1f}%, {:.2f}, {:.2f}'.format(100.0*max_accuracy, max_advantage, best_threshold))
plt.show()

fig = plt.figure
plt.title("Original model")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
for threshold in tlist:
    in_or_out_predo = do_posterior_attack(x_targets, y_targets, query_target_modelo, threshold)
    accuracy = attack_performance(in_or_out_targets, in_or_out_predo)[0]
    
    plt.plot(threshold,accuracy,marker = '+', linestyle = '-')
    
    if accuracy > max_accuracy and threshold !=0:
        max_accuracy, max_advantage, _ = attack_performance(in_or_out_targets, in_or_out_predo)
        best_threshold = threshold
print('original posterior attack accuracy, advantage, best threshold: {:.1f}%, {:.2f}, {:.2f}'.format(100.0*max_accuracy, max_advantage, best_threshold))
plt.show()