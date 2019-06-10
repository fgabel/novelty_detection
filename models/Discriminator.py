from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal, Zeros, Constant
from tensorflow.keras.layers import Input, add
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import InputSpec, Concatenate


class Discriminator():
    def __init__(self, output_classes=OUTPUT_CLASSES, fcn=False, upsampling=False, alpha=1, imagenet_filepath=None,
                 model_filepath=None):
        self.name = "Discriminator"
        self.output_classes = output_classes
        self.fcn = fcn
        self.upsampling = upsampling
        self.alpha = alpha
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

        conv_1 = Conv2D(128, (3, 3), activation='relu', name='conv_1', padding='valid')(concat)
        pool_1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(conv_1)
        conv_2 = Conv2D(256, (3, 3), activation='relu', name='conv_2', padding='valid')(pool_1)
        pool_2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(conv_2)
        conv_3 = Conv2D(512, (3, 3), activation='relu', name='conv_3', padding='valid')(pool_2)

        out = Conv2D(2, (3, 3), name='conv_4', padding='valid')(conv_3)

        discriminator = Model(inputs=[label_input, img_input], outputs=out)
        discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
        return discriminator