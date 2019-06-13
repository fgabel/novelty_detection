from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal, Zeros, Constant
from tensorflow.keras.layers import Input, add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
import numpy as np

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

class VGG16():
    def __init__(self, output_classes=1000, fcn=False, upsampling=False, alpha=1, imagenet_filepath=None,
                 model_filepath=None):
        super().__init__()

        self.name = "VGG16"
        self.output_classes = output_classes
        self.fcn = fcn
        self.upsampling = upsampling
        self.alpha = alpha

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
                    if alpha == 1:
                        weight_values[0] = weight_values[0].transpose(3, 2, 1,
                                                                      0)  # todo just because model with alpha 1 was trained using theano backend

                weight_value_tuples.append(weight_values)

            weightFC0W = np.asarray(weight_value_tuples[13][0], dtype=np.float32)
            weightFC0b = np.asarray(weight_value_tuples[13][1], dtype=np.float32)
            weightFC0W = weightFC0W.reshape((7, 7, int(512 * alpha), int(4096 * alpha)))

            weight_value_tuples[13] = [weightFC0W, weightFC0b]

            weightFC1W = np.asarray(weight_value_tuples[14][0], dtype=np.float32)
            weightFC1b = np.asarray(weight_value_tuples[14][1], dtype=np.float32)
            weightFC1W = weightFC1W.reshape((1, 1, int(4096 * alpha), int(4096 * alpha)))

            weight_value_tuples[14] = [weightFC1W, weightFC1b]

        rgb_input = Input(shape=(None, None, 3), name="rgb_input")
        # input_shape = (1024,2048)

        conv1_1 = Conv2D(int(64 * alpha), (3, 3), activation='relu', name="conv1_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[0] if len(weight_value_tuples) > 0 else None, trainable=False,
                         padding='same')(rgb_input)
        conv1_2 = Conv2D(int(64 * alpha), (3, 3), activation='relu', name="conv1_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[1] if len(weight_value_tuples) > 0 else None, trainable=False,
                         padding='same')(conv1_1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(conv1_2)
        # shape = (512,1024)

        conv2_1 = Conv2D(int(128 * alpha), (3, 3), activation='relu', name="conv2_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[2] if len(weight_value_tuples) > 0 else None, padding='same')(
            pool1)
        conv2_2 = Conv2D(int(128 * alpha), (3, 3), activation='relu', name="conv2_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[3] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv2_1)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(conv2_2)
        # shape = (256,512)

        conv3_1 = Conv2D(int(256 * alpha), (3, 3), activation='relu', name="conv3_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[4] if len(weight_value_tuples) > 0 else None, padding='same')(
            pool2)
        conv3_2 = Conv2D(int(256 * alpha), (3, 3), activation='relu', name="conv3_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[5] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv3_1)
        conv3_3 = Conv2D(int(256 * alpha), (3, 3), activation='relu', name="conv3_3",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[6] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv3_2)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(conv3_3)
        # shape = (128,256)

        conv4_1 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv4_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[7] if len(weight_value_tuples) > 0 else None, padding='same')(
            pool3)
        conv4_2 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv4_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[8] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv4_1)
        conv4_3 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv4_3",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[9] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv4_2)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(conv4_3)
        # shape = (64,128)

        conv5_1 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv5_1",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[10] if len(weight_value_tuples) > 0 else None, padding='same')(
            pool4)
        conv5_2 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv5_2",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[11] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv5_1)
        conv5_3 = Conv2D(int(512 * alpha), (3, 3), activation='relu', name="conv5_3",
                         bias_initializer=zeros_weight_filler, kernel_initializer=xavier_weight_filler,
                         weights=weight_value_tuples[12] if len(weight_value_tuples) > 0 else None, padding='same')(
            conv5_2)
        pool5 = MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(conv5_3)
        # shape = (32,64)

        if fcn:
            # Semseg Path

            fc6 = Conv2D(int(4096 * alpha), (7, 7), activation='relu',
                         weights=weight_value_tuples[13] if len(weight_value_tuples) > 0 else None, name="fc6",
                         padding='same')(pool5)
            fc6 = Dropout(0.5)(fc6)
            fc7 = Conv2D(int(4096 * alpha), (1, 1), activation='relu',
                         weights=weight_value_tuples[14] if len(weight_value_tuples) > 0 else None, name="fc7")(fc6)
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
            fc6 = Dense(num_filters, activation='relu', name="fc6", bias_initializer=fc_bias_weight_filler,
                        kernel_initializer=xavier_weight_filler)(pool5)
            fc6 = Dropout(0.5)(fc6)
            fc7 = Dense(num_filters, activation='relu', name="fc7", bias_initializer=fc_bias_weight_filler,
                        kernel_initializer=xavier_weight_filler)(fc6)
            fc7 = Dropout(0.5)(fc7)
            output = Dense(output_classes, activation='softmax_output', name="scoring",
                           bias_initializer=fc_bias_weight_filler, kernel_initializer=xavier_weight_filler)(fc7)

        self.model = Model(inputs=rgb_input, outputs=output)

        if model_filepath:
            self.model.load_weights(model_filepath)