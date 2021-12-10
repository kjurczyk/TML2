ancestor_images: contains the images we used for the dataset distillation and for model training
distilled_images: contains the distilled images we used for model training
checking_accuracy.py: checks that the accuracy of the model survived the gzipping process
checking_accuracy_confusion.py: creates a confusion matrix of the model and plots it
put_distilled_images_in_gzip.py: takes images and gzips them with one channel, also gzips the corresponding y_data (0,1, 2...)
train_target_model_and_save_LeNet_try2.py: trains a target model and saves it
train_target_model_from_hw2_and_save_many_iterations: trains a target model with the simple architecture from HW2 (it goes through a couple of iterations) and then saves it
