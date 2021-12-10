# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:40:12 2021

@author: Katarina
"""


import gzip
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
import sklearn.metrics as metrics

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def read_mnist(images_path: str, labels_path: str):
    with gzip.open(labels_path, 'rb') as labelsFile:
        labels = np.frombuffer(labelsFile.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path,'rb') as imagesFile:
        length = len(labels)
        # Load flat 28x28 px images (784 px), and convert them to 28x28 px
        features = np.frombuffer(imagesFile.read(), dtype=np.uint8, offset=16) \
                        .reshape(length, 784) \
                        .reshape(length, 28, 28, 1)
        
    return features, labels

train = {}
test = {}
validation = {}
out = {}
attack = {}

EPOCHS = 5  # change the number of epochs
BATCH_SIZE = 8 # change the number in a batch
distilled = False
training_data_filename = 'first_1000_train' # change the filename
if distilled:
    training_data_filename = training_data_filename + '_1channel'

#train['features'], train['labels'] = read_mnist('train_images_20k.gz', 'train_labels_20k.gz')

#test['features'], test['labels'] = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
# validation['features'], validation['labels'] = read_mnist('second_1000_train_images.gz', 'second_1000_train_labels.gz',)

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
            #print("I think ", i, " is IN because the ratio is ", prob_train[i], " and threshold = ", threshold)
        else:
            in_or_out_pred[i] = 0
            #print("I think ", i, " is OUT because the ratio is ", prob_train[i])

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

with gzip.open(dirPathAncestor + files[0], 'rb') as lbpath:   # training
    train['labels'] = np.frombuffer(lbpath.read(), np.uint8)
    print(train['labels'].shape)
    # print(labels_from_first_100)

with gzip.open(dirPathAncestor + files[1], 'rb') as imgpath:  # training
    train['features'] = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(train['labels']), 28, 28).reshape(len(train['labels']), 784).reshape(len(train['labels']), 28, 28, 1)
    print(f"original distilled images type: {train['features'].shape}")
    # train['features'] = np.array(tf.image.rgb_to_grayscale(train['features']))
    # print(f"shape of distilled images: {train['features'].shape}")
    # train['features'] = train['features'].reshape(len(train['labels']), 784).reshape(len(train['labels']), 28, 28, 1)
    #print(first_1000_train_images.shape)
    #print(first_1000_train_images)
    # img = Image.open("image_file_path") #for example image size : 28x28x3
    # img1 = img.convert('L')  #convert a gray scale
    
with gzip.open(dirPathAncestor + files[2], 'rb') as lbpath: # attack
    attack['labels'] = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(dirPathAncestor + files[3], 'rb') as imgpath: # attack
    attack['features'] = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(attack['labels']), 28, 28).reshape(len(attack['labels']), 784).reshape(len(attack['labels']), 28, 28, 1)

# with gzip.open(dirPathAncestor + files[4], 'rb') as lbpath: # out
#     out['labels'] = np.frombuffer(lbpath.read(), np.uint8)

# with gzip.open(dirPathAncestor + files[5], 'rb') as imgpath: # out
#     out['features'] = np.frombuffer(
#     imgpath.read(), np.uint8).reshape(len(out['labels']), 28, 28).reshape(len(out['labels']), 784).reshape(len(out['labels']), 28, 28, 1)
    
with gzip.open(dirPathAncestor + files[6], 'rb') as lbpath: # validation
    validation['labels'] = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(dirPathAncestor + files[7], 'rb') as imgpath:    # validation
    validation['features'] = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(validation['labels']), 28, 28).reshape(len(validation['labels']), 784).reshape(len(validation['labels']), 28, 28, 1)
    
with gzip.open(dirPathAncestor + files[8], 'rb') as lbpath:   # out
    out['labels'] = np.frombuffer(lbpath.read(), np.uint8)
    #print(t10k_labels_idx1_ubyte.shape)

with gzip.open(dirPathAncestor + files[9], 'rb') as imgpath:  # out
    out['features'] = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(out['labels']), 28, 28).reshape(len(out['labels']), 784).reshape(len(out['labels']), 28, 28, 1)

with gzip.open(dirPathAncestor + files[10], 'rb') as lbpath:   # test
    test['labels'] = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #print(t10k_labels_idx1_ubyte.shape)

with gzip.open(dirPathAncestor + files[11], 'rb') as imgpath:  # test
    test['features'] = np.frombuffer(
    imgpath.read(), np.uint8, offset=16).reshape(len(test['labels']), 28, 28).reshape(len(test['labels']), 784).reshape(len(test['labels']), 28, 28, 1)

print('# of training images:', train['features'].shape[0])
print('# of test images:', test['features'].shape[0])

def display_image(position):
    image = train['features'][position].squeeze()
    plt.title('Example %d. Label: %d' % (position, train['labels'][position]))
    plt.imshow(image, cmap=plt.cm.gray_r)
    
def save_model(model, base_fp):
    # save the model: first the weights then the arch
    model.save_weights('{}-weights.h5'.format(base_fp))
    with open('{}-architecture.json'.format(base_fp), 'w') as f:
        f.write(model.to_json())

    
    
train_labels_count = np.unique(train['labels'], return_counts=True)
dataframe_train_labels = pd.DataFrame({'Label':train_labels_count[0], 'Count':train_labels_count[1]})
# dataframe_train_labels



print('# of training images:', train['features'].shape[0])
print('# of validation images:', validation['features'].shape[0])


# Pad images with 0s
print(train['features'].shape)
train['features']      = np.pad(train['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
print(train['features'].shape)
validation['features'] = np.pad(validation['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
test['features']       = np.pad(test['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(train['features'][0].shape))


model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation = 'softmax'))


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])



X_train, y_train = train['features'], to_categorical(train['labels'])
X_validation, y_validation = validation['features'], to_categorical(validation['labels'])

train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)


print('# of training images:', train['features'].shape[0])
print('# of validation images:', validation['features'].shape[0])

steps_per_epoch = X_train.shape[0]//BATCH_SIZE
validation_steps = X_validation.shape[0]//BATCH_SIZE

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, 
                    validation_data=validation_generator, validation_steps=validation_steps, 
                    shuffle=True, callbacks=[tensorboard])


score = model.evaluate(test['features'], to_categorical(test['labels']))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
metrics=['accuracy'])


save_model(model,training_data_filename + "_" + str(EPOCHS) + "_" + str(BATCH_SIZE) + str(score[1]))


# query_target_model = lambda x: model.predict(x)

# # threshold = []
# # in_or_out = []
# # for i in range(0,100):
# #     threshold.append(0.01*i)
# #     #print(threshold)
# #     in_or_out_pred = do_posterior_attack(x_targets, y_targets, query_target_model, threshold=i*0.01)
    
# #     accuracy, advantage, _ = attack_performance(in_or_out_targets, in_or_out_pred)
# #     #print(accuracy)
# #     in_or_out.append(accuracy)
    
# # print(threshold)
# # print(in_or_out)

# # extract targets (some in, some out)
# x_targets, y_targets, in_or_out_targets = get_targets(train['features'], train['labels'], out['features'], out['labels'])
    
# in_or_out_pred = do_posterior_attack(x_targets, y_targets, query_target_model, threshold=0.2)
# accuracy, advantage, _ = attack_performance(in_or_out_targets, in_or_out_pred)
# print('Posterior accuracy, advantage: {:.1f}%, {:.2f}'.format(100.0*accuracy, advantage))
