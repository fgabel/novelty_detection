import tensorflow as tf
import argparse
from data_loader.data_generator import DataGenerator

from models.NoveltyGAN import NoveltyGAN

from trainers.noveltygan_trainer import NoveltyGANTrainer

from utils.config import process_config, get_config_from_json
from utils.dirs import create_dirs
from utils.logger import Logger

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

    for epoch_id in range(config.num_epochs):
        loss = trainer.train_epoch(epoch_id)

    """
    img_batch, label_batch = next(data.next_batch(10))
    pred = novelty_gan.gan.predict_on_batch(img_batch)
    print("DEBUG", pred)
    """

    # novelty_gan.gan.summary()

    if 0:
        print('_________________')
        print(cfg)
        print('_________________')


if __name__ == '__main__':

    main()
