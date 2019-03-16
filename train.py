#=================================================================================================#
# TensorFlow implementation of StyleGAN
# [A Style-Based Generator Architecture for Generative Adversarial Networks]
# (https://arxiv.org/pdf/1812.04948.pdf)
#=================================================================================================#

import tensorflow as tf
import argparse
import functools
from dataset import celeba_input_fn
from model import GAN
from network import StyleGAN
from utils import Struct

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_style_gan_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["celeba_train.tfrecord"])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--total_steps", type=int, default=1000000)
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
        growing_level=tf.cast(tf.divide(
            x=tf.train.create_global_step(),
            y=args.total_steps
        ), tf.float32),
        switching_level=tf.random_uniform([])
    )

    gan = GAN(
        generator=style_gan.generator,
        discriminator=style_gan.discriminator,
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
        hyper_params=Struct(
            generator_learning_rate=2e-3,
            generator_beta1=0.0,
            generator_beta2=0.99,
            discriminator_learning_rate=2e-3,
            discriminator_beta1=0.0,
            discriminator_beta2=0.99,
            real_gradient_penalty_weight=5.0,
            fake_gradient_penalty_weight=0.0,
        )
    )

    gan.train(
        model_dir=args.model_dir,
        total_steps=args.total_steps,
        save_checkpoint_steps=10000,
        save_summary_steps=1000,
        log_tensor_steps=1000,
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            )
        )
    )
