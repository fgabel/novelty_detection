from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import cv2
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

import tensorflow as tf

from utils.data_utils import binary_labels_to_image
from utils.data_utils import COLOR_PALETTE

"""
    Experimental BEGIN
"""

class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, tag, logs_dir):
        super().__init__()
        self.tag = tag
        self.logs_dir = logs_dir
        self.writer = tf.summary.FileWriter(self.logs_dir)

    def on_epoch_end(self, epoch, logs={}):
        generated_segmaps = logs["generated_segmaps"]
        image = logs["corresponding_image"]
        batch_size, h, w, c = generated_segmaps.shape
        seg_summaries = []

        for nr in range(batch_size):
            seg = binary_labels_to_image(generated_segmaps[nr], color_palette=COLOR_PALETTE)
            im = cv2.resize(image[nr], dsize=(256, 128), interpolation=cv2.INTER_CUBIC)
            output_real_images = 1
            if output_real_images == 1: #whether to add real images to output segmaps or not
                scaler = MinMaxScaler(feature_range=(0.01,0.99))
                im[: ,: ,0] = scaler.fit_transform(im[: ,:,0])
                im[:, :, 1] = scaler.fit_transform(im[:, :, 1])
                im[:, :, 2] = scaler.fit_transform(im[:, :, 2])
                res = np.zeros((seg.shape[0], seg.shape[1]*2, 3))
                res[:, 0:seg.shape[1], :] = seg
                res[:, seg.shape[1]:, :] = im
                plt.imsave(s, res, format='png')
            else:
                plt.imsave(s, seg, format='png')
            # Write the image to a string
            s = BytesIO()

            # Create an Image object
            seg_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=h,
                                       width=(1 + output_real_images)*w) # double width if we want real images
            # Create a Summary value
            seg_summaries.append(tf.Summary.Value(tag='seg_%s/%d' % (self.tag, nr),
                                                 image=seg_sum))

        # Create and write Summary
        seg_summary = tf.Summary(value=seg_summaries)
        self.writer.add_summary(seg_summary, epoch)

        return

"""
    Experimental END
"""


def named_logs(model, logs):
    """
    github.com/erenon
    """
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result

class NoveltyGANTrainer():
    def __init__(self, model, data, train_config):
        self.gan_model = model
        self.data = data
        self.config = train_config
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("../experiments", self.config.exp_name, "logs"),
            histogram_freq=0,
            batch_size=self.config.batch_size,
            write_graph=True,
            write_grads=True,
            write_images=True
        )
        self.modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join("../experiments", self.config.exp_name, "checkpoint/cp-{epoch:04d}.ckpt"),
            save_weights_only=True,
            period=0
        )
        self.tensorboardimage = TensorBoardImage(
            tag="Test",
            logs_dir=os.path.join("../experiments", self.config.exp_name, "summary")
        )

        self.tensorboard.set_model(self.gan_model.gan)
        self.modelcheckpoint.set_model(self.gan_model.gan)

    def train_epoch(self, id=0):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        logs = []
        for _ in loop:
            log_gan, _ = self.train_step()
            logs.append(log_gan)

        logs_avg = np.mean(logs, axis = 0)

        # self.gan_model.gan.save_weights(os.path.join("../experiments", self.config.exp_name, "checkpoint/my_model.h5"))

        self.tensorboard.on_epoch_end(id, logs=named_logs(self.gan_model.gan, logs_avg))
        # TODO: we need to keep track of validation accuracy
        self.modelcheckpoint.on_epoch_end(id)

        if 1: # print images
            img_batch, label_batch = self.data.next_batch(batch_size=1)

            generated_segmaps = self.gan_model.generator.predict_on_batch(img_batch)

            self.tensorboardimage.on_epoch_end(id, {'generated_segmaps': generated_segmaps, 'corresponding_image': img_batch})

            """
            # logdir = "../experiments/example/summary" + datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = "../experiments/example/summary"
            # Creates a file writer for the log directory.
            file_writer = tf.summary.FileWriter(logdir)
            with file_writer:
                tf.summary.image("VGG segmentation maps", generated_segmaps)
                tf.summary.image("Raw data", img_batch)
            """

        return 0

    def train_step_discriminator(self, train_on_real_data = True):
        """
        Perform a single training step for the discriminator
        We want to train the discriminator on pairs of (labels, images) with either
            labels = generator(images) or
            labels = true labels for images

        :return: The loss for this training step of the discriminator
        """

        # TODO: Add additional control parameter for ratio of fake/real data to train on.

        # We need to set the discriminator trainable first
        self.gan_model.discriminator.trainable = True

        img_batch, labels_batch = None, None
        discriminator_ground_truth = np.zeros(self.config.batch_size)

        if train_on_real_data:
            # Pick a pair of images and ground truth labels from data generator
            img_batch, labels_batch = self.data.next_batch(self.config.batch_size)
            discriminator_ground_truth.fill(0.99)
        else:
            # Let generator generate fake seg maps for another image batch
            img_batch, _ = self.data.next_batch(self.config.batch_size)
            labels_batch = self.gan_model.generator.predict(img_batch)

        discriminator_input = [labels_batch, img_batch]

        loss = self.gan_model.discriminator.train_on_batch(discriminator_input, discriminator_ground_truth)

        return loss

    def train_step_gan(self):
        """
        Perform a single training step for the GAN with frozen discriminator
        :return: The loss for this training step of the GAN
        """

        # During the training of gan,
        # the weights of discriminator should be fixed.
        # We can enforce that by setting the trainable flag
        self.gan_model.discriminator.trainable = False

        # Pick some images from TS
        # Let generator generate fake seg maps (internally) and treat them as true labels
        img_batch, _ = self.data.next_batch(self.config.batch_size)
        gan_supervision = np.ones(self.config.batch_size)

        # Train the GAN (i.e. the generator) with fixed weights of discriminator
        loss = self.gan_model.gan.train_on_batch(img_batch, gan_supervision)

        return loss

    def train_step(self, train_on_real_data=True):
        # TODO: come up with training schedule
        loss_discriminator = self.train_step_discriminator(train_on_real_data=True)
        loss_gan = self.train_step_gan()

        return loss_gan, loss_discriminator

