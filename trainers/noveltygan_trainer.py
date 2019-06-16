from tqdm import tqdm
import numpy as np

import os

import tensorflow as tf

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
            write_grads=True
        )
        self.modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join("../experiments", self.config.exp_name, "checkpoint/cp-{epoch:04d}.ckpt"),
            save_weights_only=True,
            period=0
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
        discriminator_ground_truth = np.zeros((self.config.batch_size, 2))

        if train_on_real_data:
            # Pick a pair of images and ground truth labels from data generator
            img_batch, labels_batch = next(self.data.next_batch(self.config.batch_size))
            discriminator_ground_truth[:, 0] = 0.99
        else:
            # Let generator generate fake seg maps for another image batch
            img_batch, _ = next(self.data.next_batch(self.config.batch_size))
            labels_batch = self.gan_model.generator.predict(img_batch)
            discriminator_ground_truth[:, 1] = 0.99

        discriminator_input = [labels_batch, img_batch]

        loss = self.gan_model.discriminator.train_on_batch(discriminator_input, discriminator_ground_truth)

        return loss

    def train_step_gan(self):
        """
        Perform a single training step for the GAN with frozen discriminator
        :return: The logs (loss + other metrics) for this training step of the GAN
        """

        # During the training of gan,
        # the weights of discriminator should be fixed.
        # We can enforce that by setting the trainable flag
        self.gan_model.discriminator.trainable = False

        # Pick some images from TS
        # Let generator generate fake seg maps (internally) and treat them as true labels
        img_batch, _ = next(self.data.next_batch(self.config.batch_size))
        gan_supervision = np.zeros((self.config.batch_size, 2))
        gan_supervision[:, 0] = 0.99

        # Train the GAN (i.e. the generator) with fixed weights of discriminator
        logs = self.gan_model.gan.train_on_batch(img_batch, gan_supervision)

        return logs

    def train_step(self, train_on_real_data=True):
        # TODO: come up with training schedule
        loss_discriminator = self.train_step_discriminator(train_on_real_data=True)
        loss_gan = self.train_step_gan()

        return loss_gan, loss_discriminator

