import argparse
import skimage.transform
import skimage.io
import numpy as np
def load_image(dataset_type, data_name, dataset_folders, images_directory_name, images_suffix):
    folder_images_path = dataset_folders[dataset_type] + "/" + images_directory_name

    image_filename = folder_images_path + "/" + data_name + images_suffix
    print(image_filename)
    image = skimage.io.imread(image_filename).astype(np.float32)

    return image

def load_labels(dataset_type, data_name):
    folder_labels_path = DATASET_FOLDERS[dataset_type] + "/" + CLASSES_DIRECTORY_NAME

    labels_filename = folder_labels_path + "/" + data_name + CLASSES_SUFFIX

    labels = skimage.io.imread(labels_filename).astype(np.int8)

    return labels

def get_image_data(dataset_type, data_name, dataset_folders, images_directory_name, images_suffix, data_mean, data_std):
    return preprocess_image_data(load_image(dataset_type, data_name, dataset_folders, images_directory_name, images_suffix),
                                 data_mean, data_std)

# We downsample the labels for faster training instead of performing an upsampling at the end
def get_label_data(dataset_type, data_name):
    labels = load_labels(dataset_type, data_name)

    downsampled_labels = skimage.transform.resize(labels, (int(labels.shape[0] / 8), labels.shape[1] / 8), order=0, preserve_range=True, mode='constant').astype(np.int8)

    return downsampled_labels

def preprocess_image_data(image, data_mean, data_std):
    image[:, :, 0] = (image[:, :, 0] - data_mean[0]) / data_std[0]
    image[:, :, 1] = (image[:, :, 1] - data_mean[1]) / data_std[1]
    image[:, :, 2] = (image[:, :, 2] - data_mean[2]) / data_std[2]

    return image


def binarize_labels(labels):

    height = labels.shape[0]
    width = labels.shape[1]

    labels = labels.flatten()

    labels_binary = np.zeros((labels.shape[0], OUTPUT_CLASSES + 1), dtype="bool")
    labels_binary[np.arange(labels.shape[0]), labels] = 1
    labels_binary[labels == OUTPUT_CLASSES] = 0
    labels_binary = labels_binary[:, :OUTPUT_CLASSES]

    return labels_binary.reshape(height,width,OUTPUT_CLASSES)

def get_binary_label_data(dataset_type, data_name):
    return binarize_labels(get_label_data(dataset_type, data_name))