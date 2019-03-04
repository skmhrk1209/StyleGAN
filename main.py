#=================================================================================================#
# StyleGAN: TensorFlow implementation of "A Style-Based Generator Architecture for Generative Adversarial Networks"
# [A Style-Based Generator Architecture for Generative Adversarial Networks]
# (https://arxiv.org/pdf/1812.04948.pdf)
#=================================================================================================#

import tensorflow as tf
import numpy as np
import argparse
import functools
import pickle
from dataset import celeba_input_fn
from model import GANSynth
from network import StyleGAN
from param import Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="gan_synth_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["nsynth_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--train", action="store_true")
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with open("pitch_counts.pickle", "rb") as file:
    pitch_counts = pickle.load(file)

with tf.Graph().as_default():

    tf.set_random_seed(0)

    stylegan = StyleGAN(
        min_resolution=[1, 8],
        max_resolution=[128, 1024],
        min_channels=16,
        max_channels=256,
        apply_spectral_norm=True
    )

    gan_synth = GANSynth(
        discriminator=stylegan.discriminator,
        generator=stylegan.generator,
        real_input_fn=functools.partial(
            celeba_input_fn,
            filenames=args.filenames,
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True
        ),
        fake_input_fn=lambda: (
            tf.random_normal([args.batch_size, 256]),
            tf.one_hot(tf.reshape(tf.random.multinomial(
                logits=tf.log([tf.cast(list(zip(*sorted(pitch_counts.items())))[1], tf.float32)]),
                num_samples=args.batch_size
            ), [args.batch_size]), len(pitch_counts))
        ),
        hyper_params=Param(
            discriminator_learning_rate=4e-4,
            discriminator_beta1=0.0,
            discriminator_beta2=0.9,
            generator_learning_rate=2e-4,
            generator_beta1=0.0,
            generator_beta2=0.9
        ),
        name=args.model_dir
    )

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=args.gpu,
            allow_growth=True
        )
    )

    with tf.Session(config=config) as session:

        gan_synth.initialize()
        gan_synth.train(args.total_steps)
