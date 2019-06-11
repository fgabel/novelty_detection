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


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


fcn_iou_function = K.function([vgg16.model.get_layer("rgb_input").input, K.learning_phase()],
                              [vgg16.model.get_layer("softmax_output").output])


def calculate_confusion_matrix():
    # 19 output classes
    bins = np.arange(-0.5, OUTPUT_CLASSES, 1)
    confusion_matrix = np.zeros([OUTPUT_CLASSES, OUTPUT_CLASSES], dtype=np.longlong)

    for i in range(len(validation_data_name_list)):
        image_data_volume_batch, label_data_batch = next(validation_dataset_generator)

        pred_label_data_batch = fcn_iou_function([image_data_volume_batch, 0])[0][0]
        gt_label_data = label_data_batch[0].flatten()

        pred_label_data = np.argmax(pred_label_data_batch, 2).flatten()

        cM, a, b = np.histogram2d(pred_label_data, gt_label_data, bins=bins)

        confusion_matrix = confusion_matrix + np.asarray(cM, dtype=np.longlong)

    print(confusion_matrix.shape)

    return confusion_matrix

def normalize_confusion_matrix(confusion_matrix):
    normalized_confusion_matrix = np.zeros(confusion_matrix.shape, dtype=np.float)
    sum_over_columns = np.sum(confusion_matrix, 0)
    sum_over_columns = np.maximum(sum_over_columns, 1)
    for i in range(OUTPUT_CLASSES):
        normalized_confusion_matrix[:, i] = confusion_matrix[:, i].astype(np.float) / float(sum_over_columns[i])
    return normalized_confusion_matrix

def evaluate_confusion_matrix( confusion_matrix):
	class_TP = np.zeros([OUTPUT_CLASSES], dtype=np.int64)
	class_FP = np.zeros([OUTPUT_CLASSES], dtype=np.int64)
	class_TN = np.zeros([OUTPUT_CLASSES], dtype=np.int64)
	class_FN = np.zeros([OUTPUT_CLASSES], dtype=np.int64)
	pixels = np.sum(confusion_matrix.flatten())
	for idx in range(OUTPUT_CLASSES):
		class_TP[idx] = confusion_matrix[idx, idx]
		class_FN[idx] = np.sum(confusion_matrix[:, idx]) - class_TP[idx]
		class_FP[idx] = np.sum(confusion_matrix[idx, :]) - class_TP[idx]
		class_TN[idx] = pixels - (class_TP[idx] + class_FN[idx] + class_FP[idx])

	class_IoU = []
	class_F1 = []
	class_TPR = []
	class_TNR = []
	for i in range(OUTPUT_CLASSES):
		if class_TP[i] == 0:
			class_IoU.append(0.0)
			class_F1.append(0.0)
			class_TPR.append(0.0)
		else:
			class_IoU.append(float(class_TP[i]) / float((class_TP[i] + class_FN[i] + class_FP[i])))
			class_F1.append(2 * class_TP[i] / float(2 * class_TP[i] + class_FP[i] + class_FN[i]))
			class_TPR.append(class_TP[i] / float(class_TP[i] + class_FN[i]))
		if class_TN[i] == 0:
			class_TNR.append(0.0)
		else:
			class_TNR.append(class_TN[i] / float(class_TN[i] + class_FP[i]))

	pixel_ACC = np.sum(class_TP) / float(pixels)
	mean_ACC = np.mean(class_TP / (class_TP + class_FN).astype(np.float))
	overall_IoU = np.mean(class_IoU)
	return pixel_ACC, mean_ACC, overall_IoU, class_IoU, class_F1, class_TPR, class_TNR
