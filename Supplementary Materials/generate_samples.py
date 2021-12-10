import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import gzip
import struct

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

idx = np.argsort(train_labels)
x_train_sorted = train_images[idx]
y_train_sorted = train_labels[idx]

#first 1000: from 0-100
#second 1000: from 100-200
#third 1000: from 200 - 300
#fourth 1000: from 300-400

#20000 samples: 1000-3000
start = 1000
end = 3000

x_test_zeros = train_images[train_labels == 0]
sample_zeroes_list = list(x_test_zeros)
sample_zeroes = sample_zeroes_list[start:end]
# print(sample_zeroes)
# print(len(sample_zeroes))

x_train_ones = train_images[train_labels == 1]
sample_ones_list = list(x_train_ones)
sample_ones = sample_ones_list[start:end]

x_train_twos = train_images[train_labels == 2]
sample_twos_list = list(x_train_twos)
sample_twos = sample_twos_list[start:end]

x_train_threes = train_images[train_labels == 3]
sample_threes_list = list(x_train_threes)
sample_threes = sample_threes_list[start:end]

x_train_fours = train_images[train_labels == 4]
sample_fours_list = list(x_train_fours)
sample_fours = sample_fours_list[start:end]

x_train_fives = train_images[train_labels == 5]
sample_fives_list = list(x_train_fives)
sample_fives = sample_fives_list[start:end]

x_train_sixes = train_images[train_labels == 6]
sample_sixes_list = list(x_train_sixes)
sample_sixes = sample_sixes_list[start:end]

x_train_sevens = train_images[train_labels == 7]
sample_sevens_list = list(x_train_sevens)
sample_sevens = sample_sevens_list[start:end]

x_train_eights = train_images[train_labels == 8]
sample_eights_list = list(x_train_eights)
sample_eights = sample_eights_list[start:end]

x_train_nines = train_images[train_labels == 9]
sample_nines_list = list(x_train_nines)
sample_nines = sample_nines_list[start:end]

########################################################



sampled_train_data = []
sampled_train_data = sample_zeroes + sample_ones + sample_twos + sample_threes + sample_fours + sample_fives + sample_sixes + sample_sevens + sample_eights + sample_nines

sampled_train_data_new = np.asarray(sampled_train_data)


len_samples = end-start

sampled_train_labels = []
sampled_train_labels_zeroes = list(np.zeros(len_samples, dtype = np.uint8))
sampled_train_labels_ones = list(np.ones(len_samples, dtype = np.uint8))
sampled_train_labels_twos = list(np.full(len_samples, 2, dtype = np.uint8))
sampled_train_labels_threes = list(np.full(len_samples, 3, dtype = np.uint8))
sampled_train_labels_fours = list(np.full(len_samples, 4, dtype = np.uint8))
sampled_train_labels_fives = list(np.full(len_samples, 5, dtype = np.uint8))
sampled_train_labels_sixes = list(np.full(len_samples, 6, dtype = np.uint8))
sampled_train_labels_sevens = list(np.full(len_samples, 7, dtype = np.uint8))
sampled_train_labels_eights = list(np.full(len_samples, 8, dtype = np.uint8))
sampled_train_labels_nines = list(np.full(len_samples, 9, dtype = np.uint8))

sampled_train_labels = sampled_train_labels_zeroes + sampled_train_labels_ones + sampled_train_labels_twos + sampled_train_labels_threes + sampled_train_labels_fours + sampled_train_labels_fives + sampled_train_labels_sixes + sampled_train_labels_sevens + sampled_train_labels_eights + sampled_train_labels_nines

sampled_train_labels_new = np.asarray(sampled_train_labels)
# print(sampled_train_labels_new.shape)

fd = gzip.open('train_labels_20k.gz','wb')
fd.write(sampled_train_labels_new)
fd.close()

fd = gzip.open('train_images_20k.gz','wb')
fd.write(sampled_train_data_new)
fd.close()


# image = sampled_train_data_new[250]# plot the sample
# fig = plt.figure
# plt.imshow(image, cmap='gray')
# plt.show()