# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 08:10:34 2021

@author: Katarina
"""

"""
## Simple fully-connected classifier for MNIST
"""

import keras
import matplotlib.pyplot as plt
import gzip
import numpy as np
from PIL import Image
import tensorflow as tf

"""
Get all the files 
"""
files = [
      'train_labels_20k.gz', 'train_images_20k.gz',
      'second_1000_train_labels.gz', 'second_1000_train_images.gz',
      'third_1000_train_labels.gz', 'third_1000_train_images.gz',
      'out_labels_20k.gz', 'out_images_20k.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
      ]

with gzip.open(files[0], 'rb') as lbpath:
    labels_from_first_100 = np.frombuffer(lbpath.read(), np.uint8)
    print(labels_from_first_100.shape)
    print(labels_from_first_100)

with gzip.open(files[1], 'rb') as imgpath:
    distilled_images_from_first_100 = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(labels_from_first_100), 28, 28)
    print(f"original distilled images type: {type(distilled_images_from_first_100)}")
    # distilled_images_from_first_100 = np.array(tf.image.rgb_to_grayscale(distilled_images_from_first_100))
    # print(f"type of distilled images: {type(distilled_images_from_first_100)}")
    #print(first_1000_train_images.shape)
    #print(first_1000_train_images)
    # img = Image.open("image_file_path") #for example image size : 28x28x3
    # img1 = img.convert('L')  #convert a gray scale
    
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
    


"""
gets the simple classifier
"""
def compile_model(flat_size = 28 * 28, num_labels = 10, num_hidden = 300, verbose = False, l2_regularization_constant = 0.025):
    # this is a simple feedforward neural network architecture for classification, it takes inputs of shape (flat_size,)
    # it has a hidden layer of 'num_hidden' neurons
    # finally there is softmax layer to output some probabilities over the class labels
    model = keras.models.Sequential()

    kernel_regularizer = None
    if l2_regularization_constant > 0:
        kernel_regularizer = keras.regularizers.l2(l2_regularization_constant)
    model.add(keras.layers.Dense(units=num_hidden, activation='sigmoid', input_shape=(flat_size,), kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dense(units=num_labels, activation='softmax', kernel_regularizer=kernel_regularizer))

    if verbose:
        model.summary()

    return model


"""
## More complex fully-connected classifier for MNIST with several hidden layers of varying size. 
## (Note the use of ReLU as activation function.)
"""
def get_deeper_classifier(flat_size=28*28, num_labels=10, num_hidden=[128, 96, 64, 32], verbose=True):

    # this is a deeper feedforward neural network architecture for classification, it takes inputs of shape (flat_size,)
    # it has several hidden layers with 'num_hidden' neurons
    # finally there is softmax layer to output some probabilities over the class labels
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(num_hidden[0], activation='sigmoid', input_shape=(flat_size,)))
    for s in num_hidden[1:]:
        model.add(keras.layers.Dense(units=s, activation='relu'))
    model.add(keras.layers.Dense(units=num_labels, activation='softmax'))

    if verbose:
        model.summary()

    return model



"""
## Trains the model given 'x_train' and 'y_train'
"""
# changed batch size to 1
#def train_model(model, x_train, y_train, x_test, y_test, num_epochs, batch_size=1, optimalg='sgd', lossfn='categorical_crossentropy', metrics_list=['accuracy'], verbose=False):
def train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_val, y_val, batch_size=64, optimalg='sgd', lossfn='categorical_crossentropy', metrics_list=['accuracy'], verbose=False):
    if x_test is None or y_test is None:
        verbose = False

    model.compile(optimizer=optimalg, loss=lossfn, metrics=metrics_list)
    #history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose, validation_split=.1)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=verbose, validation_data = (x_val, y_val) )

    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=verbose)

    if x_test is None or y_test is None:
        return train_loss, train_accuracy, None, None

    # print(f"x_train_shape: {x_train.shape}")
    # print(f"x_test.shape: {x_test.shape}")
    # for row in x_test:
    #     x_test_longer = np.concatenate((x_test[:], x_test[:], x_test[:]))
   
        
    # print(f"x_test_longer.shape: {x_test_longer.shape}")
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=verbose)

    if verbose:
        fig = plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Testing'], loc='best')
        plt.show()

        print('Model was trained for {} epochs. Train accuracy: {:.1f}%, Test accuracy: {:.1f}%'.format(num_epochs, train_accuracy*100.0, test_accuracy*100.0))
        #print(history.history.keys())

       
        # plt.plot(history.history['loss'])
        # #plt.plot(epochs, loss_val, 'b', label='Validation loss')
        # plt.plot(history.history['val_loss'])
        # plt.title('Training and Validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend(['Training', 'Testing'], loc='best')
        # plt.show()

    #model.save("target_model_30epochs_300hl_025rc.h5")
    return train_loss, train_accuracy, test_loss, test_accuracy


"""
## Extract 'sz' targets from in/out data
"""
def get_targets(x_in, y_in, x_out, y_out, sz=500):

    print(f"x_in.shape: {x_in.shape}")    
    x_out = x_out + x_out + x_out
    
    print(f"x_out.shape: {x_out.shape}")  

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
## Save model to file
"""
def save_model(model, base_fp):
    # save the model: first the weights then the arch
    model.save_weights('{}-weights.h5'.format(base_fp))
    with open('{}-architecture.json'.format(base_fp), 'w') as f:
        f.write(model.to_json())


model = compile_model() # compile the target model
train_in_out_size = 1000 # size of the training dataset 
num_epochs = 30    # number of epochs

# load the dataset
#train, out, test, aux = load_preprocess_mnist_data(train_in_out_size=2*train_in_out_size)

#def load_preprocess_mnist_data(train_in_out_size=2000):
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = distilled_images_from_first_100 #[:1000] # first 1000 images from .gz images file
y_train = labels_from_first_100 # first 1000 labels from .gz file

x_test = t10k_images_idx3_ubyte # all 10,000 images from test.gz?
y_test = t10k_labels_idx1_ubyte # all 10,000 labels from test.gz?

x_attack_train = third_1000_train_images # some images from mnist training data
y_attack_train = third_1000_train_labels

x_out = fourth_1000_train_images
y_out = fourth_1000_train_labels

target_train_size = x_train.shape[0]

# convert the x_test images to 3d ones so that we can test it properly
# let's do that by copying the 1000x28x28 onto 3 different layers to end up with a shape of 1000x28x28x3
#x_test = np.append(x_test, x_test)
# print(f"x_test shape is: {x_test.shape}")
# for im in x_test:
#     im = tf.image.grayscale_to_rgb(tf.convert_to_tensor(im.reshape((*im.shape,1)))) 
#     print(f"im shape: {im.shape}")
# print(f"the new x_test shape: {x_test.shape}")

# .gz has overall shape (2000, 28, 28) -- 2k images, each is 28x28 pixels
print('Loaded .gz data; shape: {} [y: {}], test shape: {} [y: {}]'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
# original
#print('Loaded .gz data; shape: {} [y: {}], test shape: {} [y: {}]'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
# Let's flatten the images for easier processing (labels don't change)
flat_vector_size = 28 * 28
flat_vector_size_distilled = 28 * 28 * 3
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
#x_targets, y_targets, in_or_out_targets = get_targets(x_train, y_train, x_out, y_out)


#train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, verbose=True)
train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_attack_train, y_attack_train, verbose=True)
   
print('Trained target model on {} records. Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size, 100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))


save_model(model,"30e_025rc_300h")


# fig = plt.figure()
# plt.plot(count, train_loss_arr)
# plt.plot(count, test_loss_arr)
# plt.ylabel('Loss')
# plt.xlabel('Regularization constant')
# plt.title('Training and Validation loss')
# plt.legend(['Training', 'Testing'], loc='best')
# plt.show()


# fig = plt.figure()
# plt.plot(count, train_accuracy_arr)
# plt.plot(count, test_accuracy_arr)

# plt.ylabel('Accuracy')
# plt.xlabel('Regularization constant')
# plt.title('Training and Validation accuracy')

# plt.legend(['Training', 'Testing'], loc='best')
# plt.show()
