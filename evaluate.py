import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import metrics
from dataset import celeba_input_fn
from network import StyleGAN

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="celeba_style_gan_model")
parser.add_argument('--filenames', type=str, nargs="+", default=["celeba_test.tfrecord"])
parser.add_argument("--batch_size", type=int, default=16)
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

    real_images = celeba_input_fn(
        filenames=args.filenames,
        batch_size=args.batch_size,
        num_epochs=1,
        shuffle=False,
        image_size=[256, 256]
    )
    fake_images = style_gan.generator(
        tf.random_normal([args.batch_size, 512]),
        tf.random_normal([args.batch_size, 512])
    )

    inception = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
    image_size = hub.get_expected_image_size(inception)

    real_images = tf.transpose(real_images, [0, 2, 3, 1])
    real_images = tf.image.resize_images(real_images, image_size)

    fake_images = tf.transpose(fake_images, [0, 2, 3, 1])
    fake_images = tf.image.resize_images(fake_images, image_size)

    real_features = inception(real_images)
    fake_features = inception(fake_images)

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

        def generator():
            while True:
                try:
                    yield session.run([real_features, fake_features])
                except tf.errors.OutOfRangeError:
                    break

        real_features, fake_features = map(np.concatenate, zip(*generator()))

        tf.logging.info("frechet_inception_distance: {}".format(
            metrics.frechet_inception_distance(real_features, fake_features)
        ))
