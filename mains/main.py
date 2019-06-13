import tensorflow as tf
import argparse
from data_loader.data_generator import DataGenerator

from models.NoveltyGAN import NoveltyGAN

from trainers.noveltygan_trainer import NoveltyGANTrainer

from trainers.noveltygan_trainer import NoveltyGANTrainer

from utils.config import process_config, get_config_from_json
from utils.dirs import create_dirs
from utils.logger import Logger

# from utils.utils import get_args
import os


parser = argparse.ArgumentParser()
parser.add_argument('--config_folder', help='The absolute path to the configs folder', default = '../configs/')
args = parser.parse_args()

import numpy as np


def main():
    # capture the config path from the run arguments
    # then process the json configuration file

    # args = get_args()

    config = process_config(args.config_folder+"train_config.json")

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create your data generator
    cfg = get_config_from_json(args.config_folder+"data_config.json")[0]
    data = DataGenerator(cfg)

    IMAGENET_FILEPATH = os.path.join(cfg.BASE_PATH, cfg.IMAGENET_FILEPATH)
    MODEL_FILEPATH = os.path.join(cfg.BASE_PATH, cfg.MODEL_FILEPATH)

    # create novelty GAN
    novelty_gan = NoveltyGAN(generator_output_classes=cfg.OUTPUT_CLASSES, fcn=True, upsampling=False, alpha=0.25,
                             imagenet_filepath=IMAGENET_FILEPATH, model_filepath=MODEL_FILEPATH)

    # create the trainer object
    trainer = NoveltyGANTrainer(novelty_gan, data, config)

    # loss = trainer.train_step_gan()
    # loss = trainer.train_step(train_on_real_data=True)
    # print("loss = ", loss)

    trainer.train_epoch()

    # Sanity check: works.
    """
    for batch in data.next_batch(3):
        labels_batch = novelty_gan.generator.predict_on_batch(batch[0])
        print("Image from batch:", batch[0].shape)
        print("Ground truth labels:", batch[1].shape)
        print("Predicted labels:", labels_batch.shape)
        # isReal = novelty_gan.discriminator.predict_on_batch([batch[1], batch[0]])
        # isReal = novelty_gan.discriminator.predict_on_batch([labels_batch, batch[0]])
        isReal = novelty_gan.gan.predict_on_batch(batch[0])
        print("discriminator prediction for ground-truth:", isReal.shape)
    """

    # Yet another sanity check for training.
    """
    for batch in data.next_batch(3):
        img_batch = batch[0]
        labels_batch = batch[1]
        # predicted_labels_batch = novelty_gan.generator.predict_on_batch(img_batch)
        target = np.zeros((3, 2))
        target[:, 0] = 1
        loss = novelty_gan.discriminator.train_on_batch([labels_batch, img_batch], target)
        print(loss)
    """

    if 0:
        print('_________________')
        print(cfg)
        print('_________________')
if __name__ == '__main__':

    main()
