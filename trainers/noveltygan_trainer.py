from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import cv2
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K

from utils.data_utils import binary_labels_to_image, softmax_output_to_binary_labels
from utils.data_utils import COLOR_PALETTE
from utils.utils import calculate_confusion_matrix, normalize_confusion_matrix, evaluate_confusion_matrix

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
            s = BytesIO()
            output_real_images = 1
            if output_real_images == 1:  # whether to add real images to output segmaps or not
                scaler = MinMaxScaler(feature_range=(0.01, 0.99))
                im[:, :, 0] = scaler.fit_transform(im[:, :, 0])
                im[:, :, 1] = scaler.fit_transform(im[:, :, 1])
                im[:, :, 2] = scaler.fit_transform(im[:, :, 2])
                res = np.zeros((seg.shape[0], seg.shape[1] * 2, 3))
                res[:, 0:seg.shape[1], :] = seg
                res[:, seg.shape[1]:, :] = im
                plt.imsave(s, res, format='png')
            else:
                plt.imsave(s, seg, format='png')
            # Write the image to a string

            # Create an Image object
            seg_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=h,
                                       width=(1 + output_real_images) * w)  # double width if we want real images
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


def named_logs(model, logs, metrics_dict = None):
    """
    github.com/erenon
    metrics_dict: an optional argument that entails metrics that are to be used optionally
    """
    result = {}
    for i in range(len(model.metrics_names)):
        result["train" + model.metrics_names[i]] = logs[i]
        # result[model.metrics_names[1]] = logs[1]
    if metrics_dict:
        result.update(metrics_dict)
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

        print("DEBUG:", self.gan_model.gan.metrics_names)

    def train_epoch(self, id=0):
        logs = []
        #train_loss_dt = []
        #train_loss_df = []
        train_loss_mixed = []
        metrics_dict = dict()

        n_per_epoch = self.config.num_iter_per_epoch
        n_discriminator_only = np.floor(n_per_epoch / 2).astype(int)
        loop1 = tqdm(range(n_discriminator_only))
        loop2 = tqdm(range(n_discriminator_only, n_per_epoch))

        for _ in loop2:
            #log_gan, train_loss_discriminator_true, train_loss_discriminator_fake = self.train_step()
            train_loss_discriminator_mixed = self.train_step(train_gan=False)
            train_loss_mixed.append(train_loss_discriminator_mixed)

        for _ in loop2:
            #log_gan, train_loss_discriminator_true, train_loss_discriminator_fake = self.train_step()
            log_gan, train_loss_discriminator_mixed = self.train_step()
            logs.append(log_gan)
            #train_loss_dt.append(train_loss_discriminator_true)
            #train_loss_df.append(train_loss_discriminator_fake)
            train_loss_mixed.append(train_loss_discriminator_mixed)

        logs_avg = np.mean(logs, axis=0)
        #metrics_dict['train loss discriminator on true data'] = np.mean(train_loss_dt)
        #metrics_dict['train loss discriminator on fake data'] = np.mean(train_loss_df)
        metrics_dict['train loss discriminator on mixed data'] = np.mean(train_loss_mixed)
        #self.gan_model.gan.save_weights(os.path.join("../experiments", self.config.exp_name, "checkpoint/my_model.h5"))
        #self.modelcheckpoint.on_epoch_end(id)

        if 0:  # print images
            img_batch, label_batch = self.data.next_batch(batch_size=1, mode="validation")

            generated_segmaps = self.gan_model.generator.predict_on_batch(img_batch)

            self.tensorboardimage.on_epoch_end(id, {
                'generated_segmaps': softmax_output_to_binary_labels(generated_segmaps),
                'corresponding_image': img_batch
            })

        img_batch, label_batch = self.data.next_batch(batch_size=5,
                                                      mode="validation")

        def evaluation_loop():
            """This function evaluates both the discriminator and the generator after each epoch"""
            # Discriminator evaluation on fake data
            # print("Fake data: ")
            dis_fake = self.gan_model.gan.evaluate(img_batch, [label_batch, np.ones(5)])
            """
            Note: self.gan_model.gan.metrics_names [
                'loss',
                'generator_loss',
                'discriminator_loss',
                'generator_acc',
                'discriminator_acc'
            ]
            """
            metrics_dict['discriminator_fake_loss'] = dis_fake[2]
            metrics_dict['discriminator_fake_acc'] = dis_fake[4]

            # Discriminator evaluation on real data
            # print("Real data: ")
            dis_real = self.gan_model.discriminator.evaluate([label_batch, img_batch], np.zeros(5))
            metrics_dict['discriminator_real_loss'] = dis_real

            if 1:
                fcn_iou_function = K.function([self.gan_model.generator.get_layer("rgb_input").input, K.learning_phase()],
                    [self.gan_model.generator.get_layer("softmax_output").output])
                pred_batch = fcn_iou_function([img_batch, 0])[0]

                # Generator evaluation in terms of IoU and stuff # TODO does not work yet
                #pred_batch = self.gan_model.generator(img_batch)  # (5, 128, 256, 19)


                confusion_matrix = calculate_confusion_matrix(pred_batch, label_batch)
                eval_out ={}
                eval_out['confMatrix'] = confusion_matrix

                eval_out['norm_confMatrix'] = normalize_confusion_matrix(confusion_matrix)

                [pixel_ACC, mean_ACC, overall_IoU, class_IoU, class_F1, class_TPR,
                 class_TNR] = evaluate_confusion_matrix(confusion_matrix)
                metrics_dict['validation IoU'] = np.mean(class_IoU)

        evaluation_loop()

        self.tensorboard.on_epoch_end(id, logs=named_logs(self.gan_model.gan, logs_avg, metrics_dict))

        return 0

    def train_step_discriminator(self, train_mode="true_data"):
        """
        Perform a single training step for the discriminator
        We want to train the discriminator on pairs of (labels, images) with either
            labels = generator(images) or
            labels = true labels for images

        :return: The loss for this training step of the discriminator
        """

        # TODO: Add additional control parameter for ratio of fake/real data to train on.

        # We need to set the discriminator trainable first
        # self.gan_model.discriminator.trainable = True

        img_batch, labels_batch = None, None
        discriminator_ground_truth = np.zeros(self.config.batch_size)

        if train_mode == "true_data":
            # Pick a pair of images and ground truth labels from data generator
            img_batch, labels_batch = self.data.next_batch(self.config.batch_size, mode="training")
            discriminator_ground_truth.fill(0.99)
        elif train_mode == "fake_data":
            # Let generator generate fake seg maps for another image batch
            img_batch, _ = self.data.next_batch(self.config.batch_size, mode="training")
            labels_batch = self.gan_model.generator.predict(img_batch)
        elif train_mode == "mixed":
            img_batch_1, labels_batch_1 = self.data.next_batch(self.config.batch_size, mode="training")
            discriminator_ground_truth_1 = np.ones(self.config.batch_size)
            img_batch_2, _ = self.data.next_batch(self.config.batch_size, mode="training")
            labels_batch_2 = self.gan_model.generator.predict(img_batch_2)
            discriminator_ground_truth_2 = np.zeros(self.config.batch_size)
            img_batch = np.concatenate((img_batch_1, img_batch_2), axis = 0)
            labels_batch = np.concatenate((labels_batch_1, labels_batch_2), axis = 0)
            discriminator_ground_truth = np.concatenate((discriminator_ground_truth_1, discriminator_ground_truth_2))
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
        # self.gan_model.discriminator.trainable = False

        # Pick some images from TS
        # Let generator generate fake seg maps (internally) and treat them as true labels
        img_batch, lbl_batch = self.data.next_batch(self.config.batch_size, mode="training")
        gan_supervision = np.ones(self.config.batch_size)

        # Train the GAN (i.e. the generator) with fixed weights of discriminator
        loss = self.gan_model.gan.train_on_batch(img_batch, [lbl_batch, gan_supervision])

        return loss

    def train_step(self, train_gan = True):
        # TODO: come up with training schedule
        #train_loss_discriminator_true = self.train_step_discriminator(train_mode="true_data")
        #train_loss_discriminator_false = self.train_step_discriminator(train_mode="fake_data")

        train_loss_discriminator_mixed = self.train_step_discriminator(train_mode="mixed")

        if train_gan:
            """
            for _ in range(3):
                train_loss_gan = self.train_step_gan()
            # print("DEBUG:", train_loss_gan)
            """
            train_loss_gan = self.train_step_gan()
            return train_loss_gan, train_loss_discriminator_mixed

        return train_loss_discriminator_mixed
        #return train_loss_gan, train_loss_discriminator_true, train_loss_discriminator_false
