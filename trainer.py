
import h5py
import numpy as np

import tensorflow as tf

import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# Set VRAM requirement to a low value but allow growing
# Important for having multiple trainings running on a single GPU
tf_config = tf.ConfigProto()

tf_config.gpu_options.per_process_gpu_memory_fraction=0.2
tf_config.gpu_options.allow_growth=True

tf_config.gpu_options.visible_device_list = "0"

#from keras.backend.tensorflow_backend import set_session
#set_session(tf.Session(config=tf_config))

# In my case - Official Cityscapes Label Scheme is different
LABEL_SCHEME = ["BICYCLE", "BUILDING", "BUS", "CAR", "FENCE", "MOTORCYCLE", "PERSON", "POLE", "RIDER", "ROAD", "SIDEWALK", "SKY", "TERRAIN", "TRAFFIC_LIGHT", "TRAFFIC_SIGN", "TRAIN", "TRUCK", "VEGETATION", "WALL"]

# Cityscapes dataset stats
DATA_MEAN = [73.1574881705, 82.9080206596, 72.3900075701]
DATA_STD = [44.906197822, 46.1445214188, 45.3104437099]

OUTPUT_CLASSES=19

IMAGES_DIRECTORY_NAME = "images"
CLASSES_DIRECTORY_NAME = "classes"

IMAGES_SUFFIX = "_leftImg8bit.png"
CLASSES_SUFFIX = "_gtFine_color.png"

DATASET_FOLDERS = {
    "training": "/export/home/ffeldman/FRANKTHETANK/data/leftImg8bit/train",
    "validation": "/export/home/ffeldman/FRANKTHETANK/data/leftImg8bit/val"
}

MODEL_FILEPATH = "/export/home/ffeldman/FRANKTHETANK/novelty_detection_pixelwise/models/gurke.h5"
IMAGENET_FILEPATH = "C:/Users/Frank/Desktop/students/students/ganovelty/models/imagenet_weights_0_25.h5"

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

import random

import skimage.io
import skimage.transform
import numpy as np

# Batch size 1 is fine for VGG
BATCH_SIZE = 1

# Naive data generator
# Can be extended to allow more features, e.g. crops, augmentation
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

import os
import numpy as np

# 500 is the total validation size, we take a subset (size 5)
number_of_validation_images = 50

validation_folder_images_path = DATASET_FOLDERS["validation"] + "/" + IMAGES_DIRECTORY_NAME
validation_folder_classes_path = DATASET_FOLDERS["validation"] + "/" + CLASSES_DIRECTORY_NAME

validation_data_name_list = []
for validation_image_path in os.listdir(validation_folder_images_path):
    # print('val image path = ', validation_image_path)
    validation_data_name = validation_image_path[0:len(validation_image_path) - len(IMAGES_SUFFIX)]
    validation_data_name_list.append(validation_data_name)

validation_data_name_list = validation_data_name_list[:number_of_validation_images]
validation_dataset_generator = data_generator("validation", validation_data_name_list)

training_folder_images_path = DATASET_FOLDERS["training"] + "/" + IMAGES_DIRECTORY_NAME
training_folder_classes_path = DATASET_FOLDERS["training"] + "/" + CLASSES_DIRECTORY_NAME

training_data_name_list = []
for training_image_path in os.listdir(training_folder_images_path):
    training_data_name = training_image_path[0:len(training_image_path) - len(IMAGES_SUFFIX)]
    training_data_name_list.append(training_data_name)

training_dataset_generator = data_generator("training", training_data_name_list, binary_labels=True)

# Numerically stable softmax - Old code probably also available in keras api already
class Softmax4D(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def compute_output_shape(self, input_shape):
        return input_shape

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal, Zeros, Constant
from tensorflow.keras.layers import Input, add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout

class VGG16():
    def __init__(self, output_classes=1000, fcn=False, upsampling=False, alpha=1, imagenet_filepath=None, model_filepath=None):
        super().__init__()

        self.name = "VGG16"
        self.output_classes = output_classes
        self.fcn = fcn
        self.upsampling = upsampling
        self.alpha = alpha

        # What are weight value tuples?
        weight_value_tuples = []

        if fcn:
            xavier_weight_filler = 'glorot_uniform'
            zeros_weight_filler = 'zeros'
            fc_bias_weight_filler = 'zeros'
        else:
            xavier_weight_filler = glorot_normal()
            zeros_weight_filler = Zeros()
            fc_bias_weight_filler = Constant(value=0.1)

        if fcn and imagenet_filepath:
            weights_of_pretrained_model = h5py.File(imagenet_filepath, mode='r')

            if 'layer_names' not in weights_of_pretrained_model.attrs and 'model_weights' in weights_of_pretrained_model:
                weights_of_pretrained_model = weights_of_pretrained_model['model_weights']
            
            # Get all the layer names
            layer_names = [encoded_layer_name.decode('utf8') for encoded_layer_name in weights_of_pretrained_model.attrs['layer_names']]
            # ... and filter those names of layers actually containg weights
            filtered_layer_names_owning_weights = []
            for layer_name in layer_names:
                weights = weights_of_pretrained_model[layer_name]
                weight_names = [encoded_layer_name.decode('utf8') for encoded_layer_name in weights.attrs['weight_names']]
                if len(weight_names):
                    filtered_layer_names_owning_weights.append(layer_name)
            layer_names = filtered_layer_names_owning_weights
            # ... now layer_names = all names of layers actually containg weights
            # Iterate over layer_names as we want to figure out all weights
            for i, layer_name in enumerate(layer_names):
                weights = weights_of_pretrained_model[layer_name]
                weight_names = [encoded_layer_name.decode('utf8') for encoded_layer_name in weights.attrs['weight_names']]
                weight_values = [weights[weight_name] for weight_name in weight_names]
                # What does weight_values[0] contain?
                weight_values[0] = np.asarray(weight_values[0], dtype=np.float32)
                # Why do we check wether weight_values[0] is of shape (*,*,*,*)?
                if len(weight_values[0].shape) == 4:
                    weight_values[0] = weight_values[0] # ???
                    if alpha == 1:
                        weight_values[0] = weight_values[0].transpose(3, 2, 1, 0) # todo just because model with alpha 1 was trained using theano backend

                weight_value_tuples.append(weight_values)

            # print(len(weight_value_tuples))
            # print(weight_value_tuples[3][0].shape)
                
            # A single entry of weight_value_tuples consists of two parts
            # weight_value_tuples[0]: the weights of shape (height, width, channels, num_filters)
            # Literature: 13 and 14 refer to the last to fully-connecte layers of VGG16, am I right?
            weightFC0W = np.asarray(weight_value_tuples[13][0], dtype=np.float32)
            weightFC0b = np.asarray(weight_value_tuples[13][1], dtype=np.float32)
            weightFC0W = weightFC0W.reshape((7, 7, int(512 * alpha), int(4096 * alpha)))

            # Is this some sort of conversion from theano to keras?
            
            weight_value_tuples[13] = [weightFC0W, weightFC0b]

            weightFC1W = np.asarray(weight_value_tuples[14][0], dtype=np.float32)
            weightFC1b = np.asarray(weight_value_tuples[14][1], dtype=np.float32)
            weightFC1W = weightFC1W.reshape((1, 1, int(4096 * alpha), int(4096 * alpha)))

            weight_value_tuples[14] = [weightFC1W, weightFC1b]

        # Here we start defining the actual architecture, right?
        # Literature: conv1_1, conv1_2, pool1, conv2_1, conv2_2, pool1, conv3_1, conv3_2, conv3_3, pool3, ...
        # ... conv4_1, conv4_2, conv4_3, pool4, conv5_1, conv5_2, conv5_3, pool5, dense, dense, dense
        
        rgb_input = Input(shape=(None, None, 3), name="rgb_input")
        # input_shape = (1024,2048)

        # Looks like weight_values_tuples[i] are bascially the weights corresponding to the i-th layer.
        conv1_1 = Conv2D(int(64 * alpha), (3, 3), activation='relu', name="conv1_1", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[0] if len(weight_value_tuples) > 0 else None, trainable=False, padding='same')(rgb_input)
        conv1_2 = Conv2D(int(64 * alpha), (3, 3), activation='relu', name="conv1_2", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[1] if len(weight_value_tuples) > 0 else None, trainable=False, padding='same')(conv1_1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(conv1_2)
        # shape = (512,1024)

        conv2_1 = Conv2D(int(128 * alpha), (3, 3), activation='relu', name="conv2_1", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[2] if len(weight_value_tuples) > 0 else None, padding='same')(pool1)
        conv2_2 = Conv2D(int(128 * alpha), (3, 3), activation='relu', name="conv2_2", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[3] if len(weight_value_tuples) > 0 else None, padding='same')(conv2_1)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(conv2_2)
        # shape = (256,512)

        conv3_1 = Conv2D(int(256 * alpha), (3, 3), activation='relu', name="conv3_1", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[4] if len(weight_value_tuples) > 0 else None, padding='same')(pool2)
        conv3_2 = Conv2D(int(256 * alpha), (3, 3), activation='relu', name="conv3_2", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[5] if len(weight_value_tuples) > 0 else None, padding='same')(conv3_1)
        conv3_3 = Conv2D(int(256 * alpha), (3, 3), activation='relu', name="conv3_3", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[6] if len(weight_value_tuples) > 0 else None, padding='same')(conv3_2)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(conv3_3)
        # shape = (128,256)

        conv4_1 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv4_1", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[7] if len(weight_value_tuples) > 0 else None, padding='same')(pool3)
        conv4_2 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv4_2", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[8] if len(weight_value_tuples) > 0 else None, padding='same')(conv4_1)
        conv4_3 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv4_3", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[9] if len(weight_value_tuples) > 0 else None, padding='same')(conv4_2)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(conv4_3)
        # shape = (64,128)

        conv5_1 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv5_1", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[10] if len(weight_value_tuples) > 0 else None, padding='same')(pool4)
        conv5_2 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv5_2", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[11] if len(weight_value_tuples) > 0 else None, padding='same')(conv5_1)
        conv5_3 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv5_3", bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler, weights=weight_value_tuples[12] if len(weight_value_tuples) > 0 else None, padding='same')(conv5_2)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(conv5_3)
        # shape = (32,64)

        if fcn:
            # Semseg Path
            
            fc6 = Conv2D(int(4096 * alpha), (7, 7), activation='relu', weights=weight_value_tuples[13] if len(weight_value_tuples) > 0 else None, name="fc6", padding='same')(pool5)
            fc6 = Dropout(0.5)(fc6)
            fc7 = Conv2D(int(4096 * alpha), (1, 1), activation='relu', weights=weight_value_tuples[14] if len(weight_value_tuples) > 0 else None, name="fc7")(fc6)
            fc7 = Dropout(0.5)(fc7)

            score_fr = Conv2D(output_classes, (1, 1), activation='relu', name="score_fr")(fc7)
            score_pool4 = Conv2D(output_classes, (1, 1), activation='relu', name="score_pool4")(pool4)
            score_pool3 = Conv2D(output_classes, (1, 1), activation='relu', name="score_pool3")(pool3)

            upsampling1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(score_fr)
            fuse_pool4 = add([upsampling1, score_pool4])
            # shape = (64,128)

            upsampling2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(fuse_pool4)
            fuse_pool3 = add([upsampling2, score_pool3])
            # shape = (128,256)

            if upsampling:
                # upsampling3 = UpSampling2DBilinear(size=(8, 8))(fuse_pool3)
                # or
                upsampling3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(fuse_pool3)
                upsampling3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(upsampling3)
                upsampling3 = UpSampling2D(size=(2, 2), interpolation='bilinear')(upsampling3)
                # shape = (1024,2048)
                output_layer = upsampling3
            else:
                output_layer = fuse_pool3
                # shape = (128,256)

            output = Softmax4D(axis=3, name="softmax_output")(output_layer)
        else:
            # Univariate Classification Path (Imagenet Pretraining)
            
            pool5 = Flatten()(pool5)
            fc6 = Dense(num_filters, activation='relu', name="fc6", bias_initializer=fc_bias_weight_filler, kernel_initializer=xavier_weight_filler)(pool5)
            fc6 = Dropout(0.5)(fc6)
            fc7 = Dense(num_filters, activation='relu', name="fc7", bias_initializer=fc_bias_weight_filler, kernel_initializer=xavier_weight_filler)(fc6)
            fc7 = Dropout(0.5)(fc7)
            output = Dense(output_classes, activation='softmax_output', name="scoring", bias_initializer=fc_bias_weight_filler, kernel_initializer=xavier_weight_filler)(fc7)

        self.model = Model(inputs=rgb_input, outputs=output)

        if model_filepath:
            self.model.load_weights(model_filepath)

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputSpec, Concatenate

from tensorflow.contrib import distributions

def adam_optimizer():
    return Adam(lr=0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-6)

def create_generator():
    vgg16 = VGG16(output_classes=OUTPUT_CLASSES, fcn=True, upsampling=False, alpha=0.25, imagenet_filepath=None, model_filepath=MODEL_FILEPATH)
    generator= vgg16.model
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

def create_discriminator():
    # Discriminator receives two inputs: label map and image
    # Note: we are using channel last convention
    label_input = Input(shape=(None, None, OUTPUT_CLASSES))
    img_input = Input(shape=(None, None, 3))
    
    # Left branch
    conv_left_1 = Conv2D(64, (5, 5), activation='relu', name='conv_left_1', padding='same')(label_input)
    # (For non-upsampled label map) (128, 256) -> (128, 256)
    
    """
    # Right branch
    conv_right_1 = Conv2D(16, (5,5), activation='relu', name='conv_right_1', padding='same')(img_input)
    # (1024, 2048) -> (1024, 2048)
    pool_right_1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_right_1')(conv_right_1)
    # (1024, 2048) -> (512, 1024)
    conv_right_2 = Conv2D(64, (5,5), activation='relu', name='conv_right_2', padding='same')(pool_right_1)
    # (512, 1024) -> (512, 1024)
    pool_right_2 = MaxPooling2D((2,2), strides=(2,2), name='pool_right_2')(conv_right_2)
    # (512, 1024) -> (256, 512)
    # TODO: There is some mismatch in dimensions
    """
    
    # Right branch
    # Note: Added one more stack of conv+relu+pool to get dimensions right
    conv_right_1 = Conv2D(4, (5,5), activation='relu', name='conv_right_1', padding='same')(img_input)
    # (1024, 2048) -> (1024, 2048)
    pool_right_1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_right_1')(conv_right_1)
    # (1024, 2048) -> (512, 1024)
    conv_right_2 = Conv2D(16, (5,5), activation='relu', name='conv_right_2', padding='same')(pool_right_1)
    # (512, 1024) -> (512, 1024)
    pool_right_2 = MaxPooling2D((2,2), strides=(2,2), name='pool_right_2')(conv_right_2)
    # (512, 1024) -> (256, 512)
    conv_right_3 = Conv2D(64, (5,5), activation='relu', name='conv_right_3', padding='same')(pool_right_2)
    # (256, 512) -> (256, 512)
    pool_right_3 = MaxPooling2D((2,2), strides=(2,2), name='pool_right_3')(conv_right_3)
    # (256, 512) -> (128, 256)
    
    # Merge the outputs of the two branches together
    concat = Concatenate(axis=-1)([conv_left_1, pool_right_3])
    # Concat now has 64*2 = 128 channels
    
    conv_1 = Conv2D(128, (3,3), activation='relu', name='conv_1', padding='valid')(concat)
    pool_1 = MaxPooling2D((2,2), strides=(2,2), name='pool_1')(conv_1)
    conv_2 = Conv2D(256, (3,3), activation='relu', name='conv_2', padding='valid')(pool_1)
    pool_2 = MaxPooling2D((2,2), strides=(2,2), name='pool_2')(conv_2)
    conv_3 = Conv2D(512, (3,3), activation='relu', name='conv_3', padding='valid')(pool_2)
    
    out = Conv2D(2, (3,3), name='conv_4', padding='valid')(conv_3)
    
    discriminator = Model(inputs=[label_input, img_input], outputs=out)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator

def create_gan(discriminator, generator):
    discriminator.trainable = False
    label_input = Input(shape=(None, None, OUTPUT_CLASSES))
    img_input = Input(shape=(None, None, 3))
    x = generator(img_input)
    # Note: we treat the output of the generator for img_input as input to discriminator
    gan_output = discriminator(x, img_input)
    gan = Model(inputs=[label_input, img_input], outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return gan

def training(epochs=1, batch_size=128):
    # TODO: How do I get X_train, y_train from the training_dataset_generator?
    # X_train, y_train = training_dataset_generator
    batch_count = X_train.shape[0] / batch_size
    
    # Creating GAN
    generator= create_generator()
    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)
    
    for e in range(1, epochs + 1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
            # Assume X_train is the test data of size (N, H, W, C)
            # First we want to grap some images according to batch_size
            # This is going to be the input to initialize the generator
            
            image_batch_false = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
            
            # Next, we generate the fake seg maps for the input X
            
            labels_batch_false = generator.predict(image_batch_false)
            
            # We want to train the discriminator on pairs (labels, images) with either
            # labels = generator(images) or
            # labels = true labels for images
            # Therefore, we now pick some pairs (labels, images) from TS
            
            image_batch_true = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
            labels_batch_true = y_train[np.random.randint(low=0,high=y_train.shape[0],size=batch_size)]
            
            # Concatenate real and generated data = input to discriminator
            discriminator_input = np.concatenate([[labels_batch_false, image_batch_false],
                                [labels_batch_true, labels_batch_true]])
            
            # We of course need to specify the ground truth for the data, i.e. wether true or fake data
            discriminator_ground_truth = np.zeros(2 * batch_size)
            discriminator_ground_truth[:batch_size] = 0.99
            
            # Pre train discriminator on fake and real data before starting the gan. 
            discriminator.trainable = True
            discriminator.train_on_batch(discriminator_input, discriminator_ground_truth)
            
            # Pick again some images from TS
            # Let generator generate fake seg maps and treat them as true labels
            image_batch = X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
            labels_batch_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            # We can enforce that by setting the trainable flag
            discriminator.trainable=False
            # Train the GAN (i.e. the generator) with fixed weights of discriminator
            # TODO: save weights using callback
            gan.train_on_batch(image_batch, labels_batch_gen)

g = create_generator()
# g.summary()
d = create_discriminator()
# d.summary()
gan = create_gan(d, g)
# gan.summary()
training(20, 128)

# Train Quarter VGG
vgg16 = VGG16(output_classes=OUTPUT_CLASSES, fcn=True, upsampling=False, alpha=0.25, imagenet_filepath=None, model_filepath=MODEL_FILEPATH)

from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr=0.0001, beta_1=0.99, beta_2=0.999, epsilon=1e-6)

loss = {}
loss_weights = {}

loss["softmax_output"] = "categorical_crossentropy"
loss_weights["softmax_output"] = 1.

vgg16.model.compile(
    optimizer=optimizer,
    loss=loss,
    loss_weights=loss_weights
)

vgg16.model.fit_generator(training_dataset_generator, steps_per_epoch=len(training_data_name_list), epochs=50, callbacks=[], validation_data=None, validation_steps=0, workers=1, use_multiprocessing=False)
