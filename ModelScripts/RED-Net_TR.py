import tensorflow as tf
import numpy as np
import tensorboard as tb

def REDnet_model_fn(features, labels, mode):
    """Image Restoration Using Convolutional Auto-encoders with Symmetric Skip-Connections
    Conv & Deconv_k + ReLu: F(X) = max(0, W_k * X + B_k)
    Skip connection: F(X_1, X_2) = max(0, X_1 + X_2)
    Total of 20 or 30 layers
    """

    # (0) Input: w*h*c

    # (i) Conv + ReLu. Filter: 32x5x5 or even 64

    # (ii) Deconv + (skip connection: elementwise sum) + ReLu
    tf.contrib.layers.conv2d_transpose(
        inputs=,    ## insert correct here
        num_outputs=1,
        kernel_size=3,
        stride=1,
        padding='SAME',
        # data_format=DATA_FORMAT_NHWC,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
    )
    # TRAINING
    # (iii).1 Loss (Mean-Squared-Error)
    # (iii).2 Optimization (ADAM, base learning rate: 10^-4)
    #         for details on momentum vectors, update rule, see recommended values p.6

    # EVALUATION

    # PREDICTION

# ESTIMATOR

# TRAINING
dir_data = 'C:/Users/timru/Documents/CODE/deepMRI1/Data/'
X1 = np.load(dir_data + 'P1_X.npy')
X1 = X1[128,:,:]
Y1 = np.load(dir_data + 'P1_Y.npy')
Y1 = Y1[128,:,:]
