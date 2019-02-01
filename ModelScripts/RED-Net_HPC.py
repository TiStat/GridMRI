import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


def REDnet_model_fn (features, labels, mode):
    """Image Restoration Using Convolutional Auto-encoders with Symmetric Skip-Connections
    :reps: number of repeated application of convoltuions/deconvolutions in one level
    :filters: list of numbers of out-filters for all convs/deconvs repetition.
    the len of filters is the number of levels.
    :kernelsize: size of each filter.
    """

    filters = [2, 4, 8, 12]
    # filters = [64,128,256,512,1024]
    kernelsize = 5
    reps = 3

    # (operations definition) --------------------------------------------------
    def leveld (inputs, filters, scope):
        d = tf.contrib.layers.repeat(
            inputs=inputs,
            num_outputs=filters,
            scope=scope,
            repetitions=reps,
            layer=tf.contrib.layers.conv2d,
            kernel_size=5,
            padding='same',
            stride=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                               'trainable': False,
                               'training': mode == tf.estimator.ModeKeys.TRAIN}
        )

        return tf.layers.max_pooling2d(
            inputs=d,
            pool_size=2,
            strides=1,
            padding='valid',
        )


    def levelu (inputs, concatin, filters, scope):
        '''anzahl der elemente der filter gibt vor, wieviele ebenen entstehen,
        reps entscheided, wieviele conv/deconvs in einer ebene sind'''
        d = tf.contrib.layers.conv2d_transpose(
            inputs=inputs,
            num_outputs=filters,
            padding='valid',
            kernel_size=2,
            stride=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                               'trainable': False,
                               'training': mode == tf.estimator.ModeKeys.TRAIN}
        )

        # print(d , concatin)

        # skip connection: extracting the last tensor of the stack/repeat call.
        d = tf.concat(
            [d, tf.reshape(concatin, tf.shape(concatin)[1:])],
            axis=3,
            name='concat'
        )

        # same repeated conv on up path
        return tf.contrib.layers.repeat(
            inputs=d,
            repetitions=reps,
            layer=tf.contrib.layers.conv2d,
            num_outputs=filters,
            padding='same',
            stride=1,
            kernel_size=kernelsize,
            activation_fn=tf.nn.relu,
            scope=scope,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                               'trainable': False,
                               'training': mode == tf.estimator.ModeKeys.TRAIN}
        )

    # (downward path) ----------------------------------------------------------
    d = tf.contrib.layers.stack(
        inputs=features,
        layer=leveld,
        stack_args=filters[:-1]
    )

    # (lowest level) -----------------------------------------------------------
    d = tf.contrib.layers.repeat(
        inputs=d,
        layer=tf.contrib.layers.conv2d,
        repetitions=reps,
        stride=1,
        num_outputs=filters[-1],
        kernel_size=kernelsize,
        padding='same',
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                           'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN},
        scope='lowest'
    )

    # (upward path) ------------------------------------------------------------
    for i, v in zip(list(range(1,len(filters)))[::-1], filters[-2::-1]):
        d = levelu(
            inputs=d,
            concatin=tf.get_default_graph().get_operation_by_name(
                'Stack/leveld_{}/leveld_{}_{}/Conv2D'.format(str(i),str(i),str(reps))).outputs,
            filters=v,
            scope='stackup/level_{}'.format(str(i))
        )

    # (last layer) -------------------------------------------------------------
    residual = tf.contrib.layers.conv2d(
        inputs=d,
        kernel_size=kernelsize,
        stride=1,
        num_outputs=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                           'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN}
    )

    # variable scope access like this during session?
    #vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #for v in vars:
    #    print(v.name)

    # get all operations
    # print(tf.get_default_graph().get_operations())

    # getting the operation.
    # based on 'scope/variable/operation'.outputtensor
    # print(tf.get_default_graph().get_operation_by_name('conv2/conv2_2/Conv2D').outputs)

    # PREDICTION
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=residual + features
        )

    # TENSORBOARD
    # For both TRAIN & EVAL:
    for var in tf.trainable_variables():
        # write out all variables
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()

    # TRAINING
    # (iii).1 L1
    # (iii).2 Optimization (ADAM, base learning rate: 10^-4)
    loss = tf.losses.absolute_difference(
        labels=labels,
        predictions=residual + features)
    tf.summary.scalar("Value_Loss_Function", loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            original_optimizer = tf.train.AdamOptimizer(
                learning_rate=0.035)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(
                original_optimizer,
                clip_norm=5.0)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

    # EVALUATION
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            eval_metric_ops={
                "accuracy": tf.metrics.mean_absolute_error(
                    labels=labels,
                    predictions=residual + features)
            }
        )

    # Adam details:
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/


# (ESTIMATOR) ------------------------------------------------------------------
# root = '/home/cloud/' # for jupyter
root =
d = datetime.datetime.now()

REDnet = tf.estimator.Estimator(
    model_fn=REDnet_model_fn,
    model_dir=root + 'model/' +
              "REDnet_{}_{}_{}_{}".format(d.month, d.day, d.hour, d.minute),
    config=tf.estimator.RunConfig(save_summary_steps=2,
                                  log_step_count_steps=2)
)

print("generated REDnet_{}_{}_{}_{} Estimator".format(d.month, d.day, d.hour, d.minute))

train_data, train_labels =


# (TRAINING) -------------------------------------------------------------------
# learning with mini batch (128 images), 50 epochs
# rewrite with tf.placeholder, session.run
# https://stackoverflow.com/questions/49743838/predict-single-image-after-training-model-in-tensorflow
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_data,
    y=train_labels,
    batch_size=2,
    num_epochs=None,
    shuffle=True)

REDnet.train(input_fn=train_input_fn, steps=20)