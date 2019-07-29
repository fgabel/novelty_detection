from __future__ import division
from __future__ import print_function

import h5py

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal, Zeros, Constant
from tensorflow.keras.layers import Input, add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import InputSpec, Concatenate
from tensorflow.keras.optimizers import Adam
K.set_image_data_format('channels_last')
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

# from ops import *
# from utils import *
from utils.layer_utils import Softmax4D

# Note: this method is just placeholder by now
# TODO: add an additional moduke for handling optimizers

def adam_optimizer(learning_rate = 0.0001):
    return Adam(lr=learning_rate, beta_1=0.99, beta_2=0.999, epsilon=1e-6)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def gen_random(mode, size):
    if mode == 'normal01': return np.random.normal(0, 1, size=size)
    if mode == 'uniform_signed': return np.random.uniform(-1, 1, size=size)
    if mode == 'uniform_unsigned': return np.random.uniform(0, 1, size=size)


class NoveltyGAN():
    """ Main model class, encompassing generator and discriminator.

    Attributes:
        generator_output_classes: number of segmentation classes of the generator
        fcn: Flag for the classification head / segmentation head
        upsampling: Flag for upsampling the segmentation maps from (128, 256) to (1024,2048)
        alpha:
        imagenet_filepath: Path to pretrained imagenet model
        model_filepath: Path to checkpointed model
        use_pooling: Wether to use pooling in the discriminator
        learning_rates: dictionary containing the learning rates for the various networks
    """
    def __init__(self, generator_output_classes=1000, fcn=False, upsampling=False, alpha=1, imagenet_filepath=None,
                 model_filepath=None, use_pooling=False, learning_rates = {}):
        super().__init__()

        self.name = "NoveltyGAN"
        self.generator_output_classes = generator_output_classes
        self.fcn = fcn
        self.upsampling = upsampling
        self.alpha = alpha
        self.imagenet_filepath = imagenet_filepath
        self.model_filepath = model_filepath
        # TODO: set self.num_filters accordingly (see down below); dummy initialization by now
        self.num_filters = 32
        # Setup generator and discriminator

        self.lr_discriminator = learning_rates.get('discriminator', 0.0001)
        self.lr_generator = learning_rates.get('generator', 0.0001)
        self.lr_gan = learning_rates.get('gan', 0.0001)

        self.pixelwise_w = 64
        self.pixelwise_h = 32

        self.use_pooling = use_pooling

        if not self.use_pooling:
            self.pixelwise_w *= 4
            self.pixelwise_h *= 4

        self.generator = None
        self.discriminator = None
        self.build_generator()
        self.build_discriminator()

        # Stick generator and discriminator together to obtain the GAN
        self.gan = None
        self.build_gan()

    def build_discriminator(self):
        """ Discriminator from the paper "Semantic Segmentation using Adversarial Networks" (https://arxiv.org/pdf/1611.08408.pdf)

            It receives two inputs: a segmentation map and an actual corresponding image and outputs a probability for
            the segmentation map coming from a generator or from the ground truth.

            Returns:
                compiled discriminator model
        """


        # Note: we are using channel last convention
        label_input = Input(shape=(128, 256, self.generator_output_classes))
        img_input = Input(shape=(1024, 2048, 3))

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
        # There is some mismatch in dimensions (done)
        """

        # Right branch
        # Note: Added one more stack of conv+relu+pool to get dimensions right
        conv_right_1 = Conv2D(4, (5, 5), activation='relu', name='conv_right_1', padding='same')(img_input)
        # (1024, 2048) -> (1024, 2048)
        pool_right_1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_right_1')(conv_right_1)
        # (1024, 2048) -> (512, 1024)
        conv_right_2 = Conv2D(16, (5, 5), activation='relu', name='conv_right_2', padding='same')(pool_right_1)
        # (512, 1024) -> (512, 1024)
        pool_right_2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_right_2')(conv_right_2)
        # (512, 1024) -> (256, 512)
        conv_right_3 = Conv2D(64, (5, 5), activation='relu', name='conv_right_3', padding='same')(pool_right_2)
        # (256, 512) -> (256, 512)
        pool_right_3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_right_3')(conv_right_3)
        # (256, 512) -> (128, 256)

        # Merge the outputs of the two branches together
        concat = Concatenate(axis=-1)([conv_left_1, pool_right_3])
        # Concat now has 64*2 = 128 channels
        # i.e. (128, 256, 128)

        conv_1 = Conv2D(128, (3, 3), activation='relu', name='conv_1', padding='same')(concat)
        if self.use_pooling:
            conv_1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(conv_1)
        conv_2 = Conv2D(256, (3, 3), activation='relu', name='conv_2', padding='same')(conv_1)
        if self.use_pooling:
            conv_2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(conv_2)
        conv_3 = Conv2D(256, (3, 3), activation='relu', name='conv_3', padding='same')(conv_2)
        conv_4 = Conv2D(1, (3, 3), activation='sigmoid', name='conv_4', padding='same')(conv_3)
        # use_pooling == False: (128, 256, 1)
        # Use_pooling == True: (32, 64, 1)

        # discriminator = Model(inputs=[label_input, img_input], outputs=conv_4[:, :, :, 0], name="discriminator")

        conv_4 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, -1))(conv_4)

        discriminator = Model(inputs=[label_input, img_input], outputs=conv_4, name="discriminator")
        discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                              optimizer=adam_optimizer(self.lr_discriminator))

        self.discriminator = discriminator

    def build_generator(self):
        """ VGG 16 with both a classification and a segmentation head

            It receives as input (a batch of) images and outputs a segmentation map for each of them.

            Returns:
                compiled generator model
        """

        weight_value_tuples = []

        if self.fcn:
            xavier_weight_filler = 'glorot_uniform'
            zeros_weight_filler = 'zeros'
            fc_bias_weight_filler = 'zeros'
        else:
            xavier_weight_filler = glorot_normal()
            zeros_weight_filler = Zeros()
            fc_bias_weight_filler = Constant(value=0.1)

        if self.fcn and self.imagenet_filepath:
            weights_of_pretrained_model = h5py.File(self.imagenet_filepath, mode='r')

            if 'layer_names' not in weights_of_pretrained_model.attrs and 'model_weights' in weights_of_pretrained_model:
                weights_of_pretrained_model = weights_of_pretrained_model['model_weights']

            layer_names = [encoded_layer_name.decode('utf8') for encoded_layer_name in
                           weights_of_pretrained_model.attrs['layer_names']]
            filtered_layer_names_owning_weights = []
            for layer_name in layer_names:
                weights = weights_of_pretrained_model[layer_name]
                weight_names = [encoded_layer_name.decode('utf8') for encoded_layer_name in
                                weights.attrs['weight_names']]
                if len(weight_names):
                    filtered_layer_names_owning_weights.append(layer_name)
            layer_names = filtered_layer_names_owning_weights
            for i, layer_name in enumerate(layer_names):
                weights = weights_of_pretrained_model[layer_name]
                weight_names = [encoded_layer_name.decode('utf8') for encoded_layer_name in
                                weights.attrs['weight_names']]
                weight_values = [weights[weight_name] for weight_name in weight_names]
                weight_values[0] = np.asarray(weight_values[0], dtype=np.float32)
                if len(weight_values[0].shape) == 4:
                    weight_values[0] = weight_values[0]
                    if self.alpha == 1:
                        weight_values[0] = weight_values[0].transpose(3, 2, 1,
                                                                      0)  # todo just because model with alpha 1 was trained using theano backend

                weight_value_tuples.append(weight_values)

            weightFC0W = np.asarray(weight_value_tuples[13][0], dtype=np.float32)
            weightFC0b = np.asarray(weight_value_tuples[13][1], dtype=np.float32)
            weightFC0W = weightFC0W.reshape((7, 7, int(512 * self.alpha), int(4096 * self.alpha)))

            weight_value_tuples[13] = [weightFC0W, weightFC0b]

            weightFC1W = np.asarray(weight_value_tuples[14][0], dtype=np.float32)
            weightFC1b = np.asarray(weight_value_tuples[14][1], dtype=np.float32)
            weightFC1W = weightFC1W.reshape((1, 1, int(4096 * self.alpha), int(4096 * self.alpha)))

            weight_value_tuples[14] = [weightFC1W, weightFC1b]
        
        rgb_input = Input(shape=(None, None, 3), name="rgb_input")
        # input_shape = (1024,2048)

        conv1_1 = Conv2D(int(64 * self.alpha), (3, 3), activation='relu', name="conv1_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[0] if len(weight_value_tuples) > 0 else None, trainable=False,
                         padding='same')(rgb_input)
        conv1_2 = Conv2D(int(64 * self.alpha), (3, 3), activation='relu', name="conv1_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[1] if len(weight_value_tuples) > 0 else None, trainable=False,
                         padding='same')(conv1_1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(conv1_2)
        # shape = (512,1024)

        conv2_1 = Conv2D(int(128 * self.alpha), (3, 3), activation='relu', name="conv2_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[2] if len(weight_value_tuples) > 0 else None, padding='same')(
            pool1)
        conv2_2 = Conv2D(int(128 * self.alpha), (3, 3), activation='relu', name="conv2_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[3] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv2_1)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(conv2_2)
        # shape = (256,512)

        conv3_1 = Conv2D(int(256 * self.alpha), (3, 3), activation='relu', name="conv3_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[4] if len(weight_value_tuples) > 0 else None, padding='same')(
            pool2)
        conv3_2 = Conv2D(int(256 * self.alpha), (3, 3), activation='relu', name="conv3_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[5] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv3_1)
        conv3_3 = Conv2D(int(256 * self.alpha), (3, 3), activation='relu', name="conv3_3",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[6] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv3_2)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(conv3_3)
        # shape = (128,256)

        conv4_1 = Conv2D(int(512 * self.alpha), (3, 3), activation='relu', name="conv4_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[7] if len(weight_value_tuples) > 0 else None, padding='same')(
            pool3)
        conv4_2 = Conv2D(int(512 * self.alpha), (3, 3), activation='relu', name="conv4_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[8] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv4_1)
        conv4_3 = Conv2D(int(512 * self.alpha), (3, 3), activation='relu', name="conv4_3",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[9] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv4_2)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(conv4_3)
        # shape = (64,128)

        conv5_1 = Conv2D(int(512 * self.alpha), (3, 3), activation='relu', name="conv5_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[10] if len(weight_value_tuples) > 0 else None, padding='same')(
            pool4)
        conv5_2 = Conv2D(int(512 * self.alpha), (3, 3), activation='relu', name="conv5_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[11] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv5_1)
        conv5_3 = Conv2D(int(512 * self.alpha), (3, 3), activation='relu', name="conv5_3",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[12] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv5_2)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(conv5_3)
        # shape = (32,64)

        if self.fcn:
            # Semseg Path

            fc6 = Conv2D(int(4096 * self.alpha), (7, 7), activation='relu',
                         weights=weight_value_tuples[13] if len(weight_value_tuples) > 0 else None, name="fc6",
                         padding='same')(pool5)
            fc6 = Dropout(0.5)(fc6)
            fc7 = Conv2D(int(4096 * self.alpha), (1, 1), activation='relu',
                         weights=weight_value_tuples[14] if len(weight_value_tuples) > 0 else None, name="fc7")(fc6)
            fc7 = Dropout(0.5)(fc7)

            score_fr = Conv2D(self.generator_output_classes, (1, 1), activation='relu', name="score_fr")(fc7)
            score_pool4 = Conv2D(self.generator_output_classes, (1, 1), activation='relu', name="score_pool4")(pool4)
            score_pool3 = Conv2D(self.generator_output_classes, (1, 1), activation='relu', name="score_pool3")(pool3)

            upsampling1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(score_fr)
            fuse_pool4 = add([upsampling1, score_pool4])
            # shape = (64,128)

            upsampling2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(fuse_pool4)
            fuse_pool3 = add([upsampling2, score_pool3])
            # shape = (128,256)

            if self.upsampling:
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

            generator = Model(inputs=rgb_input, outputs=output, name="generator")

            if self.model_filepath:
                generator.load_weights(self.model_filepath)

            loss = {}
            loss_weights = {}

            loss["softmax_output"] = "categorical_crossentropy"
            loss_weights["softmax_output"] = 1.
            generator.trainable = False
            generator.compile(
                optimizer=adam_optimizer(self.lr_generator),
                loss=loss,
                loss_weights=loss_weights
            )

            self.generator = generator

        else:
            # Univariate Classification Path (Imagenet Pretraining)

            pool5 = Flatten()(pool5)
            fc6 = Dense(self.num_filters, activation='relu', name="fc6", bias_initializer=fc_bias_weight_filler,
                        kernel_initializer=xavier_weight_filler)(pool5)
            fc6 = Dropout(0.5)(fc6)
            fc7 = Dense(self.num_filters, activation='relu', name="fc7", bias_initializer=fc_bias_weight_filler,
                        kernel_initializer=xavier_weight_filler)(fc6)
            fc7 = Dropout(0.5)(fc7)
            output = Dense(self.generator_output_classes, activation='softmax_output', name="scoring",
    bias_initializer = fc_bias_weight_filler, kernel_initializer = xavier_weight_filler)(fc7)

    def build_gan(self):
        """Method to combine discriminator and generator into a GAN"""
        self.discriminator.trainable = False
        # label_input = Input(shape=(None, None, self.generator_output_classes))
        img_input = Input(shape=(None, None, 3), name="gan_input")
        x = self.generator(img_input)
        # Note: we treat the output of the generator for img_input as input to discriminator
        gan_output = self.discriminator([x, img_input])
        # gan = Model(inputs=img_input, outputs=gan_output, name="GAN")
        gan = Model(inputs=img_input, outputs=[x, gan_output], name="GAN")

        loss = {}
        loss_weights = {}
        loss["generator"] = "categorical_crossentropy"
        loss_weights["generator"] = 0.2
        loss["discriminator"] = "binary_crossentropy"
        loss_weights["discriminator"] = 0.8

        gan.compile(
            optimizer=adam_optimizer(self.lr_gan),
            loss=loss,
            loss_weights=loss_weights,
            metrics=['accuracy']
        )

        """
        gan.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=adam_optimizer(self.lr_gan),
                    metrics=['accuracy'])
        """
        self.gan = gan

    def sampler(self, z, y=None):
        pass

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, epoch, exp_name):
        self.generator.save_weights(os.path.join(
            "../experiments", exp_name, "checkpoint/generator/cp-{0:04d}.ckpt".format(epoch))
        )
        self.discriminator.save_weights(os.path.join(
            "../experiments", exp_name, "checkpoint/discriminator/cp-{0:04d}.ckpt".format(epoch))
        )

    def load(self, exp_name):
        latest_generator = tf.train.latest_checkpoint(
            os.path.join("../experiments", exp_name, "checkpoint/generator")
        )
        latest_discriminator = tf.train.latest_checkpoint(
            os.path.join("../experiments", exp_name, "checkpoint/discriminator")
        )
        self.generator.load_weights(latest_generator)
        self.discriminator.load_weights(latest_discriminator)




