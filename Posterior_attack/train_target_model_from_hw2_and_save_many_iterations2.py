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


# change these variables 

num_epochs = 30    # number of epochs - originally 30
num_hidden_layers = 300 # number of hidden layers - originally 300
regularization_constant = 0.025 # rg - originally 0.025
training_data_filename = '10_distilled_batch_size_64_5epochs' # change the filename
# training_data_filename = 'first_1000_train' # change the filename


"""
Get all the files 
"""


dirPathDistilled = "../distilled_images/"
dirPathAncestor = "../ancestor_images/"
files = [
      training_data_filename + '_1channel_labels.gz', training_data_filename +'_1channel_images.gz',
      'second_1000_attack_labels.gz', 'second_1000_attack_images.gz',
      'third_1000_out_labels.gz', 'third_1000_out_images.gz',
      'fourth_1000_val_labels.gz','fourth_1000_val_images.gz',
      'out_labels_20k.gz', 'out_images_20k.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
      ]

with gzip.open(dirPathDistilled + files[0], 'rb') as lbpath:   # train
    first_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)
    #print(first_1000_train_labels.shape)
    #print(first_1000_train_labels)

with gzip.open(dirPathDistilled + files[1], 'rb') as imgpath:  # train
    first_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(first_1000_train_labels), 28, 28)
    #print(first_1000_train_images.shape)
    #print(first_1000_train_images)
    
# files = [
#       training_data_filename + '_labels.gz', training_data_filename +'_images.gz',
#       'second_1000_attack_labels.gz', 'second_1000_attack_images.gz',
#       'third_1000_out_labels.gz', 'third_1000_out_images.gz',
#       'fourth_1000_val_labels.gz','fourth_1000_val_images.gz',
#       'out_labels_20k.gz', 'out_images_20k.gz',
#       't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
#       ]

# with gzip.open(dirPathAncestor + files[0], 'rb') as lbpath:   # train
#     first_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)
#     #print(first_1000_train_labels.shape)
#     #print(first_1000_train_labels)

# with gzip.open(dirPathAncestor + files[1], 'rb') as imgpath:  # train
#     first_1000_train_images = np.frombuffer(
#     imgpath.read(), np.uint8).reshape(len(first_1000_train_labels), 28, 28)
#     #print(first_1000_train_images.shape)
#     #print(first_1000_train_images)
    
with gzip.open(dirPathAncestor + files[2], 'rb') as lbpath: # attack
    second_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(dirPathAncestor + files[3], 'rb') as imgpath: # attack
    second_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(second_1000_train_labels), 28, 28)

with gzip.open(dirPathAncestor + files[4], 'rb') as lbpath: # out
    third_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(dirPathAncestor + files[5], 'rb') as imgpath: # out
    third_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(third_1000_train_labels), 28, 28)
    
with gzip.open(dirPathAncestor + files[6], 'rb') as lbpath: # validation
    fourth_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)

with gzip.open(dirPathAncestor + files[7], 'rb') as imgpath: # validation
    fourth_1000_train_images = np.frombuffer(
    imgpath.read(), np.uint8).reshape(len(fourth_1000_train_labels), 28, 28)
    
with gzip.open(dirPathAncestor + files[10], 'rb') as lbpath: # test
    t10k_labels_idx1_ubyte = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #print(t10k_labels_idx1_ubyte.shape)

with gzip.open(dirPathAncestor + files[11], 'rb') as imgpath: # test
    t10k_images_idx3_ubyte = np.frombuffer(
    imgpath.read(), np.uint8, offset=16).reshape(len(t10k_labels_idx1_ubyte), 28, 28)

def open_training_data(filename, distilled = False):
    if distilled:
        path_to_use = dirPathDistilled
    else:
        path_to_use = dirPathAncestor
        
    with gzip.open(path_to_use + filename + "_labels.gz", 'rb') as lbpath:   # train
        first_1000_train_labels = np.frombuffer(lbpath.read(), np.uint8)
   

    with gzip.open(path_to_use + filename + "_images.gz", 'rb') as imgpath:  # train
        first_1000_train_images = np.frombuffer(
        imgpath.read(), np.uint8).reshape(len(first_1000_train_labels), 28, 28)
        
    return first_1000_train_labels, first_1000_train_images
     

"""
gets the simple classifier
"""
def compile_model(flat_size = 28 * 28, num_labels = 10, num_hidden = num_hidden_layers, verbose = False, l2_regularization_constant = regularization_constant):
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
def train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_val, y_val, batch_size=8, optimalg='sgd', lossfn='categorical_crossentropy', metrics_list=['accuracy'], verbose=False):
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
    path_to_save = "trained_models_like_hw2/"
    #   model.save_weights('{}-weights.h5'.format(path_to_save + training_data_filename + "/" + base_fp))
    #with open('{}-architecture.json'.format(path_to_save + training_data_filename + "/" + base_fp), 'w') as f:
    model.save_weights('{}-weights.h5'.format(path_to_save  + "/" + base_fp))
    with open('{}-architecture.json'.format(path_to_save  + "/" + base_fp), 'w') as f:
        f.write(model.to_json())

def plot_chart(x_axis, y_axis, title, x_label, y_label):
    fig = plt.figure()
    plt.plot(x_axis, y_axis)
    
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    
    plt.show()


# train_in_out_size = 1000 # size of the training dataset 


# load the dataset
#train, out, test, aux = load_preprocess_mnist_data(train_in_out_size=2*train_in_out_size)

#def load_preprocess_mnist_data(train_in_out_size=2000):
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = first_1000_train_images #[:1000] # first 1000 images from .gz images file
y_train = first_1000_train_labels # first 1000 labels from .gz file

x_test = t10k_images_idx3_ubyte # all 10,000 images from test.gz?
y_test = t10k_labels_idx1_ubyte # all 10,000 labels from test.gz?

# x_attack_train = third_1000_train_images # some images from mnist training data
# y_attack_train = third_1000_train_labels

x_out = third_1000_train_images
y_out = third_1000_train_labels
# print(f"x_out shape: {x_out.shape}")
# print(f"y_out shape: {y_out.shape}")


x_validation = fourth_1000_train_images
y_validation = fourth_1000_train_labels

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
# x_attack_train = x_attack_train.reshape(x_attack_train.shape[0], flat_vector_size)
x_validation = x_validation.reshape(x_validation.shape[0], flat_vector_size)
x_out = x_out.reshape(x_out.shape[0], flat_vector_size)
print("made it here")


print(y_train)
y_train = np.arange(10.)
# y_train[0][0] = 1
# for i in range(1,10):
#     y_train[i][0] = 0
#     y_train[i][i] = 1

# Put the labels in "one-hot" encoding using keras' to_categorical()
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# y_attack_train = keras.utils.to_categorical(y_attack_train, num_classes)
y_out = keras.utils.to_categorical(y_out, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)
print(f"y test shape is: {y_train.shape}")
print(f"y_out is in the form of {y_out}")
print(f"y_train is in the form of {y_train[0]}")
print(f"y_train is in the form of {y_train[1]}")
print(f"y_train is in the form of {y_train[2]}")
print(f"y_train is in the form of {y_train[3]}")
print(f"y_train is in the form of {y_train[4]}")
print(f"y_train is in the form of {y_train[5]}")

# extract targets (some in, some out)
#x_targets, y_targets, in_or_out_targets = get_targets(x_train, y_train, x_out, y_out)

"""


x_axis = []
y_axis = []
y_axis_loss = []
best_accuracy = 0
best_nh = 0
best_rc = 0
best_epochs = 0


for nh in range(50,350, 20): # 50-350
    model = compile_model(num_hidden = nh)
    #train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, verbose=True)
    train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_validation, y_validation, verbose=False)
    x_axis.append(nh)
    y_axis.append(test_accuracy)
    y_axis_loss.append(test_loss)
    save_model(model,training_data_filename + "_epochs" + str(num_epochs) + "_hidden_layers" + str(nh) + "_rc" + str(regularization_constant))
    if test_accuracy > best_accuracy:
        best_nh = nh
    print(f"nh: {nh}")

plot_chart(x_axis, y_axis, "Number of Hidden Layers vs Accuracy", "Number of Hidden Layers", "Accuracy")
plot_chart(x_axis, y_axis_loss, "Number of Hidden Layers vs Loss", "Number of Hidden Layers", "Loss")
    
x_axis = []
y_axis = []   
y_axis_loss = []
best_accuracy = 0
for rc in np.arange(0.0,0.2,0.01): # (0.0,0.2,0.01)
    model = compile_model(l2_regularization_constant = rc)
    train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_validation, y_validation, verbose=False)
    x_axis.append(rc)
    y_axis.append(test_accuracy)
    y_axis_loss.append(test_loss)
    save_model(model,training_data_filename + "_epochs" + str(num_epochs) + "_hidden_layers" + str(num_hidden_layers) + "_rc" + str(rc))
    if test_accuracy > best_accuracy:
        best_rc = rc
    print(f"rc: {rc}")
    
plot_chart(x_axis, y_axis, "RC vs Accuracy", "Regularization Constant", "Accuracy")
plot_chart(x_axis, y_axis_loss, "RC vs Loss", "Regularization Constant", "Loss")
    
x_axis = []
y_axis = []  
y_axis_loss = []  
best_accuracy = 0
for epochs in range (0,75,10):
    model = compile_model()
    train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, epochs, x_validation, y_validation, verbose=False)
    x_axis.append(epochs)
    y_axis.append(test_accuracy)
    y_axis_loss.append(test_loss)
    save_model(model,training_data_filename + "_epochs" + str(epochs) + "_hidden_layers" + str(num_hidden_layers) + "_rc" + str(regularization_constant))
    if test_accuracy > best_accuracy:
        best_epochs = epochs
    print(f"epochs: {epochs}")
        
plot_chart(x_axis, y_axis, "Epochs vs Accuracy", "Number of Epochs", "Accuracy")
plot_chart(x_axis, y_axis_loss, "Epochs vs Loss", "Number of Epochs", "Loss")
    

model = compile_model(l2_regularization_constant = best_rc, num_hidden = best_nh)
train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, best_epochs, x_validation, y_validation, verbose=True)
print('Trained target model on {} records. Epochs: {}, NH: {}, RC: {} Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size, best_epochs, best_nh, best_rc, 100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))
save_model(model,training_data_filename + "_epochs" + str(best_epochs) + "_hidden_layers" + str(best_nh) + "_rc" + str(best_rc))


"""


"""
num_epochs = 5
model = compile_model()
train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_validation, y_validation, verbose=False)
print('Trained target model on {} records. Epochs: {}, NH: {}, RC: {} Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size, num_epochs, num_hidden_layers, regularization_constant, 100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))
save_model(model,training_data_filename + "_epochs" + str(num_epochs) + "_hidden_layers" + str(num_hidden_layers) + "_rc" + str(regularization_constant))

num_epochs = 10
train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_validation, y_validation, verbose=False)
print('Trained target model on {} records. Epochs: {}, NH: {}, RC: {} Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size, num_epochs, num_hidden_layers, regularization_constant, 100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))
save_model(model,training_data_filename + "_epochs" + str(num_epochs) + "_hidden_layers" + str(num_hidden_layers) + "_rc" + str(regularization_constant))

num_epochs = 30
train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_validation, y_validation, verbose=False)
print('Trained target model on {} records. Epochs: {}, NH: {}, RC: {} Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size, num_epochs, num_hidden_layers, regularization_constant, 100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))
save_model(model,training_data_filename + "_epochs" + str(num_epochs) + "_hidden_layers" + str(num_hidden_layers) + "_rc" + str(regularization_constant))

num_epochs = 50
train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_validation, y_validation, verbose=False)
print('Trained target model on {} records. Epochs: {}, NH: {}, RC: {} Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size, num_epochs, num_hidden_layers, regularization_constant, 100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))
save_model(model,training_data_filename + "_epochs" + str(num_epochs) + "_hidden_layers" + str(num_hidden_layers) + "_rc" + str(regularization_constant))

"""

print(y_train[9])
print(type(y_train))
print(type(y_validation))

num_epochs = 5
for i in range(0,5):
    model = compile_model()
    train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, x_train, y_train, x_test, y_test, num_epochs, x_validation, y_validation, verbose=False)
    print('Trained target model on {} records. Epochs: {}, NH: {}, RC: {} Train accuracy and loss: {:.1f}%, {:.2f} -- Test accuracy and loss: {:.1f}%, {:.2f}'.format(target_train_size, num_epochs, num_hidden_layers, regularization_constant, 100.0*train_accuracy, train_loss, 100.0*test_accuracy, test_loss))
    save_model(model,training_data_filename + "_8batchsize_epochs" + str(num_epochs) + "_fixed_hidden_layers" + str(num_hidden_layers) + "_rc" + str(regularization_constant) + "_acc" + str(test_accuracy))





# fig = plt.figure()
# plt.plot(count, train_loss_arr)
# plt.plot(count, test_loss_arr)
# plt.ylabel('Loss')
# plt.xlabel('Regularization constant')
# plt.title('Training and Validation loss')
# plt.legend(['Training', 'Testing'], loc='best')
# plt.show()


