from tqdm import tqdm
import numpy as np

import os

from tensorflow.keras.models import save_model

class NoveltyGANTrainer():
    def __init__(self, model, data, train_config):
        self.gan_model = model
        self.data = data
        self.config = train_config

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
            print('Loss: ', loss)
            # TODO: Why is the loss constant right away?
        loss = np.mean(losses)

        # TODO: Do something with the loss here (maybe try to obtain accuracy aswell)
        # ...

        # TODO: improve on saving the weights here
        self.gan_model.gan.save_weights(os.path.join("../experiments", self.config.exp_name, "checkpoint/my_model.h5"))

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

        loss = self.gan_model.discriminator.fit(x = discriminator_input,
                                      y = discriminator_ground_truth,
                                      batch_size = self.config.batch_size,
                                      callbacks = # TODO: KERAS CALLBACK
                                      )

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
        loss = self.gan_model.gan.fit(x = img_batch,
                                      y = gan_supervision,
                                      batch_size = self.config.batch_size,
                                      callbacks = # TODO: KERAS CALLBACK
                                      )

        return loss

    def train_step(self, train_on_real_data=True):
        # TODO: come up with training schedule
        loss_discriminator = self.train_step_discriminator(train_on_real_data=True)
        loss_gan = self.train_step_gan()

        return loss_gan

