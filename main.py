#=================================================================================================#
# TensorFlow implementation of StyleGAN
# [A Style-Based Generator Architecture for Generative Adversarial Networks]
# (https://arxiv.org/pdf/1812.04948.pdf)
#=================================================================================================#

import tensorflow as tf
import numpy as np
import argparse
import functools
import pickle
from dataset import celeba_input_fn
from model import GAN
from network import StyleGAN
from param import Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_style_gan_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["celeba_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--train", action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    style_gan = StyleGAN(
        min_resolution=[4, 4],
        max_resolution=[256, 256],
        min_channels=16,
        max_channels=512,
        mapping_layers=8,
        growing_level=tf.cast(tf.get_variable(
            name="global_step",
            initializer=0,
            trainable=False
        ) / args.total_steps, tf.float32),
        switching_level=tf.random_uniform([])
    )

    gan = GAN(
        discriminator=style_gan.discriminator,
        generator=style_gan.generator,
        real_input_fn=functools.partial(
            celeba_input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True,
            image_size=[256, 256]
        ),
        fake_input_fn=lambda: (
            tf.random_normal([args.batch_size, 512]),
            tf.random_normal([args.batch_size, 512])
        ),
        hyper_params=Param(
            discriminator_learning_rate=2e-3,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            generator_learning_rate=2e-3,
            generator_beta1=0.0,
            generator_beta2=0.99,
            r1_gamma=10.0,
            r2_gamma=0.0
        ),
        model_dir=args.model_dir
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )

    with tf.Session(config=config) as session:

        gan.initialize()
        gan.train(args.total_steps)
