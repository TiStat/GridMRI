import tensorflow as tf
import numpy as np
import tensorboard as tb
from tensorflow.contrib.layers.python.layers import initializers

def DnCNN_model_fn(features, labels, depth=5, filters= 10, kernelsize=3, mode= 'train'):
    """ Beyond a Gaussian Denoiser: Residual learning of Deep CNN for Image Denoising.
        Residual learning (originally with X = True + Noisy, Y = noisy)
        :depth: total number of conv-layers in network
        :filters: number of filters for every convolution"""


    # (0) Input layer. (batch)-1*256*256*1(channels)
    input_layer = tf.reshape(features['X'], [-1,256,256,1])

    # (i) Conv + Relu. Filters: 64, 3x3x1, same padding
    conv_first = tf.contrib.layers.conv2d(
        inputs = input_layer,
        num_outputs = filters,
        kernel_size = kernelsize,
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
    convbn = tf.contrib.layers.repeat(inputs=conv_first, repetitions=depth-2, layer=tf.contrib.layers.conv2d,
                                      num_outputs=filters, padding='SAME', stride=1, kernel_size=[kernelsize,kernelsize], # actually volume [k,k,filters]
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=tf.layers.batch_normalization,
                                      normalizer_params= {'momentum':0.99, 'epsilon':0.001, 'trainable': False, 'training': mode == tf.estimator.ModeKeys.TRAIN},
                                      scope='conv2')  # passed arguments to conv2d, scope variable to share variables

    # (iii) Conv. 3x3x64, same padding
    conv_last = tf.contrib.layers.conv2d(
        inputs=input_layer,
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
    # (iv).1 Loss (averaged mean-squared-error)

    # (iv).2 optimizer (SGD + momentum & Adam)

    # EVALUATION
    # (v) Metric

    # PREDICTION

# ESTIMATOR

# TRAINING:
# learning with mini batch (128 images), 50 epochs


dir_data = 'C:/Users/timru/Documents/CODE/deepMRI1/Data/'
X1 = np.load(dir_data + 'P1_X.npy')
X1 = X1[:5,:,:]
Y1 = np.load(dir_data + 'P1_Y.npy')
Y1 = Y1[:5,:,:]

# train_input_fn =

