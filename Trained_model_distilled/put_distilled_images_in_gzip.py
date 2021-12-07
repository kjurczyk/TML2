#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import random
import gzip
#import struct


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
#import os

DataDir = "distilled_first_1000_train_images"

print("[INFO] loading images...")
imagePaths = list(paths.list_images(DataDir))
data = []
labels = []
print(f"imagePaths: {imagePaths}")

count = 0

for imagePath in imagePaths:
        # extract the class label from the filename
        # label = imagePath.split(os.path.sep)[-2]
        label = count
        print(f"label: {label}")

        # # load the input image (224x224) 
        image = load_img(imagePath, target_size=(28, 28))
        image = img_to_array(image)
        #print("successfully loaded an image")
        
        # fig = plt.figure
        # plt.imshow(label, cmap='gray')
        # plt.show()
        
        
        print(f"image shape: {image.shape}")
        #image = np.squeeze(image, axis=-1)
        #print(f"image shape: {image.shape}")
        
        # # update the data and labels lists, respectively
        data.append(image)
        
        #print(image)
        #print("appended image")
        labels.append(label)
        
        count = count + 1

data = np.array(data, dtype= np.uint8)
labels = np.array(labels, dtype= np.uint8)
#print(data)





# The images
fd = gzip.open('10_distilled_images_from_first_100.gz','wb')
fd.write(data)
fd.close()

# The labels
fd = gzip.open('10_labels_from_first_100.gz','wb')
fd.write(labels)
fd.close()





# print(np.zeros(100, dtype = np.uint8))
# print(np.full(100, 2, dtype = np.uint8))
# f_out = gzip.open('sampled_train_data_new.gz', 'wt')
# print(f_out)
# f_out.writelines(sampled_train_data_new)
# print(f_out)
# f_out.close()

image = data[2]# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()