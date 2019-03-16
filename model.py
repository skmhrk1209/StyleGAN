import tensorflow as tf
from utils import Struct


class GAN(object):

    def __init__(self, generator, discriminator, real_input_fn, fake_input_fn, hyper_params):
        # =========================================================================================
        real_images = real_input_fn()
        fake_images = generator(*fake_input_fn())
        # =========================================================================================
        real_features, real_logits = discriminator(real_images)
        fake_features, fake_logits = discriminator(fake_images)
        real_logits = tf.squeeze(real_logits, axis=1)
        fake_logits = tf.squeeze(fake_logits, axis=1)
        # =========================================================================================
        # Non-Saturating + Zero-Centered Gradient Penalty
        # [Generative Adversarial Networks]
        # (https://arxiv.org/abs/1406.2661)
        # [Which Training Methods for GANs do actually Converge?]
        # (https://arxiv.org/pdf/1801.04406.pdf)
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        generator_losses = tf.nn.softplus(-fake_logits)
        # -----------------------------------------------------------------------------------------
        # non-saturating loss
        discriminator_losses = tf.nn.softplus(-real_logits)
        discriminator_losses += tf.nn.softplus(fake_logits)
        # zero-centerd gradient penalty on data distribution
        if hyper_params.real_gradient_penalty_weight:
            real_gradients = tf.gradients(real_logits, [real_images])[0]
            real_gradient_penalties = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])
            discriminator_losses += real_gradient_penalties * hyper_params.real_gradient_penalty_weight
        # zero-centerd gradient penalty on generator distribution
        if hyper_params.fake_gradient_penalty_weight:
            fake_gradients = tf.gradients(fake_logits, [fake_images])[0]
            fake_gradient_penalties = tf.reduce_sum(tf.square(fake_gradients), axis=[1, 2, 3])
            discriminator_losses += fake_gradient_penalties * hyper_params.fake_gradient_penalty_weight
        # -----------------------------------------------------------------------------------------
        # losss reduction
        generator_loss = tf.reduce_mean(generator_losses)
        discriminator_loss = tf.reduce_mean(discriminator_losses)
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
            loss=generator_loss,
            var_list=generator_variables,
            global_step=tf.train.get_or_create_global_step()
        )
        discriminator_train_op = discriminator_optimizer.minimize(
            loss=discriminator_loss,
            var_list=discriminator_variables
        )
        # =========================================================================================
        # tensors and operations used later
        self.operations = Struct(
            discriminator_train_op=discriminator_train_op,
            generator_train_op=generator_train_op
        )
        self.tensors = Struct(
            global_step=tf.train.get_global_step(),
            real_images=tf.transpose(real_images, [0, 2, 3, 1]),
            fake_images=tf.transpose(fake_images, [0, 2, 3, 1]),
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss
        )

    def train(self, model_dir, total_steps, save_checkpoint_steps, save_summary_steps, log_tensor_steps, config):

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
                    save_steps=save_summary_steps,
                    summary_op=tf.summary.merge([
                        tf.summary.scalar(name=name, tensor=tensor) if tensor.shape.ndims == 0 else
                        tf.summary.image(name=name, tensor=tensor, max_outputs=4)
                        for name, tensor in self.tensors.items()
                    ])
                ),
                tf.train.LoggingTensorHook(
                    tensors={
                        name: tensor for name, tensor in self.tensors.items()
                        if tensor.shape.ndims == 0
                    },
                    every_n_iter=log_tensor_steps,
                ),
                tf.train.StopAtStepHook(
                    last_step=total_steps
                )
            ]
        ) as session:

            while not session.should_stop():
                for name, operation in self.operations.items():
                    session.run(operation)
