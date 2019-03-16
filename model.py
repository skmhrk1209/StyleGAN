import tensorflow as tf
from utils import Struct


class GAN(object):

    def __init__(self, generator, discriminator, train_real_input_fn, train_fake_input_fn,
                 valid_real_input_fn, valid_fake_input_fn, hyper_params):
        # =========================================================================================
        train_real_images = train_real_input_fn()
        # -----------------------------------------------------------------------------------------
        valid_real_images = valid_real_input_fn()
        # =========================================================================================
        train_fake_images = generator(*train_fake_input_fn())
        # -----------------------------------------------------------------------------------------
        valid_fake_images = generator(*valid_fake_input_fn())
        # =========================================================================================
        train_real_features, train_real_logits = discriminator(train_real_images)
        train_fake_features, train_fake_logits = discriminator(train_fake_images)
        train_real_logits = tf.squeeze(train_real_logits, axis=1)
        train_fake_logits = tf.squeeze(train_fake_logits, axis=1)
        # -----------------------------------------------------------------------------------------
        valid_real_features, valid_real_logits = discriminator(valid_real_images)
        valid_fake_features, valid_fake_logits = discriminator(valid_fake_images)
        valid_real_logits = tf.squeeze(valid_real_logits, axis=1)
        valid_fake_logits = tf.squeeze(valid_fake_logits, axis=1)
        # =========================================================================================
        # Non-Saturating + Zero-Centered Gradient Penalty
        # [Generative Adversarial Networks]
        # (https://arxiv.org/abs/1406.2661)
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        train_generator_losses = tf.nn.softplus(-train_fake_logits)
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        train_discriminator_losses = tf.nn.softplus(-train_real_logits)
        train_discriminator_losses += tf.nn.softplus(train_fake_logits)
        # zero-centerd gradient penalty on data distribution
        if hyper_params.real_gradient_penalty_weight:
            train_real_gradients = tf.gradients(train_real_logits, [train_real_images])[0]
            train_real_gradient_penalties = tf.reduce_sum(tf.square(train_real_gradients), axis=[1, 2, 3])
            train_discriminator_losses += 0.5 * hyper_params.real_gradient_penalty_weight * train_real_gradient_penalties
        # zero-centerd gradient penalty on generator distribution
        if hyper_params.fake_gradient_penalty_weight:
            train_fake_gradients = tf.gradients(train_fake_logits, [train_fake_images])[0]
            train_fake_gradient_penalties = tf.reduce_sum(tf.square(train_fake_gradients), axis=[1, 2, 3])
            train_discriminator_losses += 0.5 * hyper_params.fake_gradient_penalty_weight * train_fake_gradient_penalties
        # -----------------------------------------------------------------------------------------
        # losss reduction
        train_generator_loss = tf.reduce_mean(train_generator_losses)
        train_discriminator_loss = tf.reduce_mean(train_discriminator_losses)
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        valid_generator_losses = tf.nn.softplus(-valid_fake_logits)
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        valid_discriminator_losses = tf.nn.softplus(-valid_real_logits)
        valid_discriminator_losses += tf.nn.softplus(valid_fake_logits)
        # zero-centerd gradient penalty on data distribution
        if hyper_params.real_gradient_penalty_weight:
            valid_real_gradients = tf.gradients(valid_real_logits, [valid_real_images])[0]
            valid_real_gradient_penalties = tf.reduce_sum(tf.square(valid_real_gradients), axis=[1, 2, 3])
            valid_discriminator_losses += 0.5 * hyper_params.real_gradient_penalty_weight * valid_real_gradient_penalties
        # zero-centerd gradient penalty on generator distribution
        if hyper_params.fake_gradient_penalty_weight:
            valid_fake_gradients = tf.gradients(valid_fake_logits, [valid_fake_images])[0]
            valid_fake_gradient_penalties = tf.reduce_sum(tf.square(valid_fake_gradients), axis=[1, 2, 3])
            valid_discriminator_losses += 0.5 * hyper_params.fake_gradient_penalty_weight * valid_fake_gradient_penalties
        # -----------------------------------------------------------------------------------------
        # losss reduction
        valid_generator_loss = tf.reduce_mean(valid_generator_losses)
        valid_discriminator_loss = tf.reduce_mean(valid_discriminator_losses)
        # =========================================================================================
        generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.generator_learning_rate,
            beta1=hyper_params.generator_beta1,
            beta2=hyper_params.generator_beta2
        )
        discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=hyper_params.discriminator_learning_rate,
            beta1=hyper_params.discriminator_beta1,
            beta2=hyper_params.discriminator_beta2
        )
        # -----------------------------------------------------------------------------------------
        generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
        # =========================================================================================
        generator_train_op = generator_optimizer.minimize(
            loss=train_generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=train_discriminator_loss,
            var_list=discriminator_variables
        )
        # =========================================================================================
        # frechet_inception_distance
        train_frechet_inception_distance = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(train_real_features, train_fake_features)
        # -----------------------------------------------------------------------------------------
        # frechet_inception_distance
        valid_frechet_inception_distance = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(valid_real_features, valid_fake_features)
        # =========================================================================================
        # tensors and operations used later
        self.operations = Struct(
            discriminator_train_op=discriminator_train_op,
            generator_train_op=generator_train_op
        )
        self.tensors = Struct(
            train_real_images=tf.transpose(train_real_images, [0, 2, 3, 1]),
            train_fake_images=tf.transpose(train_fake_images, [0, 2, 3, 1]),
            train_generator_loss=train_generator_loss,
            train_discriminator_loss=train_discriminator_loss,
            train_frechet_inception_distance=train_frechet_inception_distance,
            valid_real_images=tf.transpose(valid_real_images, [0, 2, 3, 1]),
            valid_fake_images=tf.transpose(valid_fake_images, [0, 2, 3, 1]),
            valid_generator_loss=valid_generator_loss,
            valid_discriminator_loss=valid_discriminator_loss,
            valid_frechet_inception_distance=valid_frechet_inception_distance
        )

    def train(self, model_dir, total_steps, save_checkpoint_steps,
              save_train_summary_steps, save_valid_summary_steps,
              log_train_tensor_steps, log_valid_tensor_steps, config):

        with tf.train.SingularMonitoredSession(
            scaffold=tf.train.Scaffold(
                init_op=tf.global_variables_initializer(),
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer()
                )
            ),
            checkpoint_dir=model_dir,
            config=config,
            hooks=[
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=model_dir,
                    save_steps=save_checkpoint_steps,
                    saver=tf.train.Saver(
                        max_to_keep=10,
                        keep_checkpoint_every_n_hours=12,
                    )
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_train_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(name=name, tensor=tensor) if tensor.shape.ndims == 0 else
                        tf.summary.image(name=name, tensor=tensor, max_outputs=4)
                        for name, tensor in self.tensors.items() if "train" in name
                    ])
                ),
                tf.train.SummarySaverHook(
                    output_dir=model_dir,
                    save_steps=save_valid_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(name=name, tensor=tensor) if tensor.shape.ndims == 0 else
                        tf.summary.image(name=name, tensor=tensor, max_outputs=4)
                        for name, tensor in self.tensors.items() if "valid" in name
                    ])
                ),
                tf.train.LoggingTensorHook(
                    tensors={
                        name: tensor for name, tensor in self.tensors.items()
                        if "train" in name and tensor.shape.ndims == 0
                    },
                    every_n_iter=log_train_tensor_steps,
                ),
                tf.train.LoggingTensorHook(
                    tensors={
                        name: tensor for name, tensor in self.tensors.items()
                        if "valid" in name and tensor.shape.ndims == 0
                    },
                    every_n_iter=log_valid_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                for name, operation in self.operations.items():
                    session.run(operation)
