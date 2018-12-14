import tensorflow as tf
import numpy as np
import tensorboard as tb
from tensorflow.contrib.layers.python.layers import initializers


def DnCNN_model_fn (features, labels, mode='train'):
    """ Beyond a Gaussian Denoiser: Residual learning of Deep CNN for Image Denoising.
        Residual learning (originally with X = True + Noisy, Y = noisy)
        :depth: total number of conv-layers in network
        :filters: number of filters for every convolution"""

    # unfortunate workaround, due to tf 1.11.0 has rigid argument checking in Estimator API
    depth = 5
    filters = 10
    kernelsize = 3

    # (0) Input layer. (batch)-1*256*256*1(channels)
    input_layer = features['X']

    # (i) Conv + Relu. Filters: 64, 3x3x1, same padding
    conv_first = tf.contrib.layers.conv2d(
        inputs=input_layer,
        num_outputs=filters,
        kernel_size=kernelsize,
        padding='SAME',
        stride=1,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        trainable=True,
        scope=None
    )

    # (ii) Conv + BN + Relu. Filters: 64, 3x3x64,  same padding
    # 17 or 20 of these layers
    conv_bn = tf.contrib.layers.repeat(inputs=conv_first, repetitions=depth - 2, layer=tf.contrib.layers.conv2d,
                                       num_outputs=filters, padding='SAME', stride=1,
                                       kernel_size=[kernelsize, kernelsize],  # actually volume [k,k,filters]
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=tf.layers.batch_normalization,
                                       normalizer_params={'momentum': 0.99, 'epsilon': 0.001, 'trainable': False,
                                                          'training': mode == tf.estimator.ModeKeys.TRAIN},
                                       scope='conv2')  # passed arguments to conv2d, scope variable to share variables

    # (iii) Conv. 3x3x64, same padding
    conv_last = tf.contrib.layers.conv2d(
        inputs=conv_bn,
        num_outputs=1,
        kernel_size=kernelsize,
        padding='SAME',
        stride=1,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        trainable=True,
        scope=None
    )

    # TRAINING
    # (iv).1 Loss (originally averaged mean-squared-error on patches)
    loss = tf.losses.mean_squared_error(
        labels=labels['Y'],
        predictions=conv_last)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=conv_last,
        loss=loss,
        train_op=tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-08).minimize(  # optimizer and objective
            loss=loss,
            global_step=tf.train.get_global_step()),
        eval_metric_ops={
            "accuracy": tf.metrics.mean_absolute_error(
                labels=labels,
                predictions=conv_last)}
    )

    # Ada(ptive) m(oment) estimation details can be found with:
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/


# ------------------------------------------------------------------------------
root = 'C:/Users/timru/Documents/CODE/deepMRI1/'

# ESTIMATOR
DnCNN = tf.estimator.Estimator(model_fn=DnCNN_model_fn,  # an EstimatorSpec object
                               model_dir=root + 'model/' + 'DnCNN_model'
                               # TODO runconfig
                               )

# TRAINING:
# learning with mini batch (128 images), 50 epochs
X1 = np.load(root + 'Data/' + 'P1_X.npy')
train_data = tf.reshape(X1[:5, :, :], [-1, 256, 256, 1])
test_data = tf.reshape(X1[5:8, :, :], [-1, 256, 256, 1])

Y1 = np.load(root + 'Data/' + 'P1_Y.npy')
train_labels = tf.reshape(Y1[:5, :, :], [-1, 256, 256, 1])
test_labels = tf.reshape(Y1[5:8, :, :], [-1, 256, 256, 1])

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": train_data},
    y={'Y': train_labels},
    batch_size=2,
    num_epochs=None,
    shuffle=True)

DnCNN.train(input_fn=train_input_fn,
            steps=20)
