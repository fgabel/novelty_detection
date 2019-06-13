from tqdm import tqdm
import numpy as np

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

        loss = np.mean(losses)

        # TODO: Do something with the loss here (maybe try to obtain accuracy aswell)

        # Serialize model weights
        # self.gam_model.save()

    def train_step_discriminator(self):
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

        # Pick a pair of images and ground truth labels from data generator
        img_batch_true, labels_batch_true = self.data.next_batch(self.config.batch_size)

        # We want to train the discriminator on both - real and fake data
        # So let generator generate fake seg maps for another image batch
        img_batch_false, _ = self.data.next_batch(self.config.batch_size)
        labels_batch_false = self.gan_model.generator.predict(img_batch_false)

        # Concatenate this data and set ground truth
        discriminator_input = np.concatenate([[labels_batch_true, img_batch_true],
                                              [labels_batch_false, img_batch_false]])
        discriminator_ground_truth = np.zeros(2 * self.config.batch_size)
        discriminator_ground_truth[:self.config.batch_size] = 0.99

        loss = self.gan_model.discriminator.train_on_batch(discriminator_input, discriminator_ground_truth)

        return loss

    def train_step_gan(self):
        """
        Perform a single training step for the GAN with frozen discriminator
        :return: The loss for this training step of the GAN
        """
        return 0

    def train_step(self):
        return 0

