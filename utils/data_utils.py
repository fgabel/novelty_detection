import argparse
import skimage.transform
import skimage.io
import numpy as np

COLOR_PALETTE = [(119, 11, 32), (70, 70, 70), (0, 60,100), ( 0, 0, 142), (190, 153, 153), (0, 0, 230),
                 (220, 20, 60), (153, 153, 153), (255, 0, 0), (128, 64, 128), (244, 35, 232), (70, 130, 180),
                 (152, 251, 152), (250, 170, 30), (220, 220, 0), (0, 80, 100), (0, 0, 70), (107, 142, 35),
                 (102,102,156)]

def load_image(dataset_type, data_name, dataset_folders, images_directory_name, images_suffix):
    folder_images_path = dataset_folders[dataset_type] + "/" + images_directory_name
    image_filename = folder_images_path + "/" + data_name + images_suffix
    image = skimage.io.imread(image_filename).astype(np.float32)

    return image

def load_labels(dataset_type, data_name, dataset_folders, classes_directory_name, classes_suffix):
    folder_labels_path = dataset_folders[dataset_type] + "/" + classes_directory_name

    labels_filename = folder_labels_path + "/" + data_name + classes_suffix

    labels = skimage.io.imread(labels_filename).astype(np.int8)

    return labels

def get_image_data(dataset_type, data_name, dataset_folders, images_directory_name, images_suffix, data_mean, data_std):
    return preprocess_image_data(load_image(dataset_type, data_name, dataset_folders, images_directory_name, images_suffix),
                                 data_mean, data_std)

# We downsample the labels for faster training instead of performing an upsampling at the end
def get_label_data(dataset_type, data_name, dataset_folders, classes_directory_name, classes_suffix):
    labels = load_labels(dataset_type, data_name, dataset_folders, classes_directory_name, classes_suffix)

    downsampled_labels = skimage.transform.resize(labels, (int(labels.shape[0] / 8), labels.shape[1] / 8), order=0, preserve_range=True, mode='constant').astype(np.int8)

    return downsampled_labels

def preprocess_image_data(image, data_mean, data_std):
    image[:, :, 0] = (image[:, :, 0] - data_mean[0]) / data_std[0]
    image[:, :, 1] = (image[:, :, 1] - data_mean[1]) / data_std[1]
    image[:, :, 2] = (image[:, :, 2] - data_mean[2]) / data_std[2]

    return image


def binarize_labels(labels, output_classes = 1000):

    height = labels.shape[0]
    width = labels.shape[1]

    labels = labels.flatten()

    labels_binary = np.zeros((labels.shape[0], output_classes + 1), dtype="bool")
    labels_binary[np.arange(labels.shape[0]), labels] = 1
    labels_binary[labels == output_classes] = 0
    labels_binary = labels_binary[:, :output_classes]

    return labels_binary.reshape(height,width,output_classes)

def get_binary_label_data(dataset_type, data_name, dataset_folders, classes_directory_name, classes_suffix, output_classes):
    # dataset_type, data_name, dataset_folders, classes_directory_name, classes_suffix
    return binarize_labels(get_label_data(
        dataset_type, data_name, dataset_folders, classes_directory_name, classes_suffix
    ), output_classes=output_classes)

def binary_labels_to_image(binary_labels, color_palette):
    """
    Creates an RGB image from a inbary label map
    :param binary_labels: A binary label map of shape (H, W, C)
    :param color_palette: A color palette corresponding to the C channels of the label map
    :return: A numpy array of shape (H, W, 3) (the resulting RGB image)
    """
    H, W, C = binary_labels.shape
    img = np.zeros((C, H, W, 3), dtype=np.float32)
    indices = binary_labels == 1
    for c in range(C):
        img[c, indices[:, :, c]] = color_palette[c]
    img /= 255.
    return np.sum(img, axis=0)