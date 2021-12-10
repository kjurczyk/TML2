Folders
ancestor_images: contains the images we used for the dataset distillation and for model training
distilled_images: contains the distilled images we used for model training
training: contains the many models we trained (.json and .h5)

Python Files
checking_accuracy.py: checks that the accuracy of the model survived the gzipping process
checking_accuracy_confusion.py: creates a confusion matrix of the model and plots it
put_distilled_images_in_gzip.py: takes images and gzips them with one channel, also gzips the corresponding y_data (0,1, 2...)
running_posterior_attack.py: runs the posterior attack on a given model
train_target_model_and_save_LeNet_try2.py: trains a target model and saves it
train_target_model_from_hw2_and_save_many_iterations: trains a target model with the simple architecture from HW2 (it goes through a couple of iterations) and then saves it
__init__.py: The python script for initilizing dataset and configure dataset settings
fashion.py: Load the fashion customized dataset and convert arrays to tensors for distillation
job.sh: Sample job script that submitted to the HiperGator
logs.zip: All the logs
results-MNIST.zip: Replicated their results on MNIST
