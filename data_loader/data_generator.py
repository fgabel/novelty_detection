import random
import os
import skimage.io
import skimage.transform
import numpy as np
from utils.data_utils import get_image_data, get_binary_label_data, get_label_data, preprocess_image_data, load_image, \
    load_labels
from utils.config import get_config_from_json

# Batch size 1 is fine for VGG
BATCH_SIZE = 1


class DataGenerator:
    """
    Class for data generation in batches(both train and validation)
    """

    def __init__(self, dataconfig):
        self.config = dataconfig
        # print(self.config.dataset_folders[mode])

        # load data here

        # print(self.data_name_list)
        # Note: we need binary label data during training!
        self.binary_labels = 1

    def next_batch(self, batch_size, mode):
        X = []  # care that this does not cause unintended consequences
        Y = []

        classes_suffix = self.config.classes_suffix
        OUTPUT_CLASSES = self.config.OUTPUT_CLASSES
        self.mode = mode
        
        self.data_name_list = get_training_data_list(self.config.dataset_folders[mode] + '/images',
                                                     self.config.images_suffix)
        batch = np.random.choice(self.data_name_list, batch_size)
        for data_name in batch:
            image_data_volume = get_image_data(self.mode, data_name, self.config.dataset_folders,
                                               self.config.images_directory_name, self.config.images_suffix,
                                               data_mean=self.config.data_mean, data_std=self.config.data_std)

            if mode == "out_of_distribution_images":
                classes_suffix = '_gtFine_labelIds.png'
                OUTPUT_CLASSES = 1
            if self.binary_labels:
                label_data = get_binary_label_data(self.mode, data_name, self.config.dataset_folders,
                                                   self.config.classes_directory_name, classes_suffix,
                                                   OUTPUT_CLASSES)
            else:
                label_data = get_label_data(self.mode, data_name, self.config.dataset_folders,
                                            self.config.classes_directory_name, classes_suffix)

            X.append(image_data_volume)
            Y.append(label_data)
            # yield np.array(X), np.array(Y)
        # Where is yield supposed to be?
        # yield np.array(X), np.array(Y)
        return np.array(X), np.array(Y)


def get_training_data_list(training_folder_images_path, images_suffix):
    """
    Return a list of training images given their suffix and a path
    :param training_folder_images_path:
    :param IMAGES_SUFFIX:
    :return:
    """
    training_data_name_list = []

    for training_image_path in os.listdir(training_folder_images_path):
        training_data_name = training_image_path[0:len(training_image_path) - len(images_suffix)]
        # print(training_data_name)
        training_data_name_list.append(training_data_name)
    return training_data_name_list


# cfg = get_config_from_json("../configs/data_config.json")[0]
"""
cfg = get_config_from_json("../configs/my_data_config.json")[0]
dat = DataGenerator(cfg)
for batch in dat.next_batch(2):
    print(batch)
"""
