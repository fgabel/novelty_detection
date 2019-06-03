import argparse
import skimage.transform
import skimage.io

def load_image(dataset_type, data_name):
    folder_images_path = DATASET_FOLDERS[dataset_type] + "/" + IMAGES_DIRECTORY_NAME

    image_filename = folder_images_path + "/" + data_name + IMAGES_SUFFIX

    image = skimage.io.imread(image_filename).astype(np.float32)

    return image

def load_labels(dataset_type, data_name):
    folder_labels_path = DATASET_FOLDERS[dataset_type] + "/" + CLASSES_DIRECTORY_NAME

    labels_filename = folder_labels_path + "/" + data_name + CLASSES_SUFFIX

    labels = skimage.io.imread(labels_filename).astype(np.int8)

    return labels

def preprocess_image_data(image):
    image[:, :, 0] = (image[:, :, 0] - DATA_MEAN[0]) / DATA_STD[0]
    image[:, :, 1] = (image[:, :, 1] - DATA_MEAN[1]) / DATA_STD[1]
    image[:, :, 2] = (image[:, :, 2] - DATA_MEAN[2]) / DATA_STD[2]

    return image

def get_image_data(dataset_type, data_name):
    return preprocess_image_data(load_image(dataset_type, data_name))

# We downsample the labels for faster training instead of performing an upsampling at the end
def get_label_data(dataset_type, data_name):
    labels = load_labels(dataset_type, data_name)

    downsampled_labels = skimage.transform.resize(labels, (int(labels.shape[0] / 8), labels.shape[1] / 8), order=0, preserve_range=True, mode='constant').astype(np.int8)

    return downsampled_labels

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

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

