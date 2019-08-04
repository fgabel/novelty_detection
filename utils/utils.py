import argparse
import skimage.transform
import skimage.io
import numpy as np
from tensorflow.math import argmax
from tensorflow.keras.backend import flatten
from .data_utils import softmax_output_to_binary_labels

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args





def calculate_confusion_matrix(pred_batch, label_batch):
    # 19 output classes
    OUTPUT_CLASSES = 19
    IoUs = []
    bins = np.arange(-0.5, OUTPUT_CLASSES, 1)
    confusion_matrix = np.zeros([OUTPUT_CLASSES, OUTPUT_CLASSES], dtype=np.longlong)
    for i in range(label_batch.shape[0]):  # iterate through batch size

        gt_label_data = np.argmax(label_batch[i], 2).flatten()

        pred_label_data = np.argmax(pred_batch[i], 2).flatten() #

        cM, a, b = np.histogram2d(pred_label_data, gt_label_data, bins=bins)

        confusion_matrix = confusion_matrix + np.asarray(cM, dtype=np.longlong)
        IoUs.append(np.sum(np.equal(pred_label_data, gt_label_data))/pred_label_data.shape)
        #print(IoUs)

    return confusion_matrix, np.mean(IoUs)

def normalize_confusion_matrix(confusion_matrix):
    normalized_confusion_matrix = np.zeros(confusion_matrix.shape, dtype=np.float)
    sum_over_columns = np.sum(confusion_matrix, 0)
    sum_over_columns = np.maximum(sum_over_columns, 1)
    for i in range(19):
        normalized_confusion_matrix[:, i] = confusion_matrix[:, i].astype(np.float) / float(sum_over_columns[i])
    return normalized_confusion_matrix

def evaluate_confusion_matrix( confusion_matrix):
	class_TP = np.zeros([19], dtype=np.int64)
	class_FP = np.zeros([19], dtype=np.int64)
	class_TN = np.zeros([19], dtype=np.int64)
	class_FN = np.zeros([19], dtype=np.int64)
	pixels = np.sum(confusion_matrix.flatten())
	for idx in range(19):
		class_TP[idx] = confusion_matrix[idx, idx]
		class_FN[idx] = np.sum(confusion_matrix[:, idx]) - class_TP[idx]
		class_FP[idx] = np.sum(confusion_matrix[idx, :]) - class_TP[idx]
		class_TN[idx] = pixels - (class_TP[idx] + class_FN[idx] + class_FP[idx])

	class_IoU = []
	class_F1 = []
	class_TPR = []
	class_TNR = []
	for i in range(19):
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