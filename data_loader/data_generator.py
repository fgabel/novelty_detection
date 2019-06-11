import random

import skimage.io
import skimage.transform
import numpy as np
from utils.data_utils import get_image_data, get_binary_label_data, get_label_data
# Batch size 1 is fine for VGG
BATCH_SIZE = 1


class DataGenerator:
    """
    Class for data generation in batches(both train and validation)
    """
    def __init__(self, dataconfig):
        self.config = dataconfig
        # load data here
        self.data_name_list = get_training_data_list(mode, self.config.datapath)

    def next_batch(self, batch_size):
        X = [] # care that this does not cause unintended consequences
        Y = []
        batch = np.random.choice(self.data_name_list, batch_size)
        for data_name in batch:
            image_data_volume = get_image_data(dataset_type, data_name)

            if binary_labels:
                label_data = get_binary_label_data(dataset_type, data_name)
            else:
                label_data = get_label_data(dataset_type, data_name)

            X.append(image_data_volume)
            Y.append(label_data)

            yield np.array(X), np.array(Y)

def get_training_data_list(training_folder_images_path, IMAGES_SUFFIX):
    """
    Return a list of training images given their suffix and a path
    :param training_folder_images_path:
    :param IMAGES_SUFFIX:
    :return:
    """
    training_data_name_list = []
    for training_image_path in os.listdir(training_folder_images_path):
        training_data_name = training_image_path[0:len(training_image_path) - len(IMAGES_SUFFIX)]
        training_data_name_list.append(training_data_name)
    return training_data_name_list

