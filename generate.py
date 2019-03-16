import tensorflow as tf
import numpy as np
import skimage
import argparse
from network import StyleGAN
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_style_gan_model")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--total_steps", type=int, default=1000000)
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():

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

    images = style_gan.generator(
        tf.random_normal([args.batch_size, 512]),
        tf.random_normal([args.batch_size, 512])
    )

    with tf.train.SingularMonitoredSession(
        scaffold=tf.train.Scaffold(
            init_op=tf.global_variables_initializer(),
            local_init_op=tf.group(
                tf.local_variables_initializer(),
                tf.tables_initializer()
            )
        ),
        checkpoint_dir=args.model_dir,
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=args.gpu,
                allow_growth=True
            )
        )
    ) as session:

        sample_dir = Path("samples")

        if not sample_dir.exists():
            sample_dir.mkdir(parents=True, exist_ok=True)

        def linear_map(inputs, in_min, in_max, out_min, out_max):
            return out_min + (inputs - in_min) / (in_max - in_min) * (out_max - out_min)

        for image in session.run(images):
            skimage.io.imsave(
                fname=sample_dir / "{}.jpg".format(len(list(sample_dir.glob("*.jpg")))),
                arr=linear_map(image[0], -1.0, 1.0, 0.0, 255.0).astype(np.uint8).clip(0, 255)
            )
