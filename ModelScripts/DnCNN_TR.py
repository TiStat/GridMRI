import tensorflow as tf
import numpy as np
import tensorboard as tb
from tensorflow.contrib.layers.python.layers import initializers


def DnCNN_model_fn(features, labels, mode):
    """ Beyond a Gaussian Denoiser: Residual learning of Deep CNN for Image Denoising.
        Residual learning (originally with X = True + Noisy, Y = noisy)"""

    # (0) Input layer. 256x256x1
    input_layer = tf.reshape(features['X'], [-1,256,256,1])

    # (i) Conv + Relu. Filters: 64, 3x3x1, same padding
    tf.contrib.layers.conv2d(
        inputs = input_layer,
        num_outputs = 32,
        kernel_size = 3,
        stride=1,
        padding='SAME',
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

    # (iii) Conv. 3x3x64, same padding

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

