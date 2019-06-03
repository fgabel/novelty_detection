import random

import skimage.io
import skimage.transform
import numpy as np

# Batch size 1 is fine for VGG
BATCH_SIZE = 1

def data_generator(dataset_type, data_name_list, binary_labels=False):
    while True:

        random.shuffle(data_name_list)

        X = []
        Y = []
        for data_name in data_name_list:

            image_data_volume = get_image_data(dataset_type, data_name)

            if binary_labels:
                label_data = get_binary_label_data(dataset_type, data_name)
            else:
                label_data = get_label_data(dataset_type, data_name)

            X.append(image_data_volume)
            Y.append(label_data)

            if len(X) == BATCH_SIZE:
                yield np.array(X), np.array(Y)
                X = []
                Y = []

# TODO: turn the above function into an appropriate DataGenerator class!


class DataGenerator:

    """

    """
    def __init__(self, config):
        self.config = config
        # load data here
        get_training_data_list
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]

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