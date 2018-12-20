import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
#tf.enable_eager_execution()
#tf.executing_eagerly()
import numpy as np
import datetime


# (MODEL) --------------------------------------------------------------------
def DnCNN_model_fn (features, labels, mode):
    """ Beyond a Gaussian Denoiser: Residual learning of Deep CNN for Image Denoising.
        Residual learning (originally with X = True + Noisy, Y = noisy)
        :depth: total number of conv-layers in network
        :filters: number of filters for every convolution"""

    # workaround, due to tf 1.11.0 rigid argument checking in Estimator
    depth = 4
    filters = 10
    kernelsize = 5

    # (0) Input layer. (batch)-1*256*256*1(channels)
    input_layer = features #['X']

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
    conv_bn = tf.contrib.layers.repeat(
        inputs=conv_first, repetitions=depth - 2, layer=tf.contrib.layers.conv2d,
        num_outputs=filters, padding='SAME', stride=1,
        kernel_size=[kernelsize, kernelsize],
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001, 'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN},
        scope='conv2'
        # passed arguments to conv2d, scope variable to share variables

        # TODO BN: updates_collections for running mean/var for predictions
        # look at tf.contrib.layers.batch_norm doc.
    )


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
        labels=labels, # passed as array not dict!
        predictions=conv_last + input_layer)

    # TENSORBOARD
    tf.summary.scalar("Value_Loss_Function", loss)

    for var in tf.trainable_variables():  # write out all variables
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=conv_last + input_layer,
        loss=loss,
        train_op=tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-08).minimize(
            loss=loss,
            global_step=tf.train.get_global_step()),
        eval_metric_ops={
            "accuracy": tf.metrics.mean_absolute_error(
                labels=labels,
                predictions=conv_last + input_layer)}
    )

    # Adam details:
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/


# (ESTIMATOR SPEC) -------------------------------------------------------------
# root = '/home/cloud/' # for jupyter
root = 'C:/Users/timru/Documents/CODE/deepMRI1/'
d = datetime.datetime.now()

DnCNN = tf.estimator.Estimator(
    model_fn=DnCNN_model_fn,
    model_dir=root + 'model/' +
              "DnCNN_{}_{}_{}_{}".format(d.month, d.day, d.hour, d.minute),
    config=tf.estimator.RunConfig(save_summary_steps=2,
                                  log_step_count_steps=10)
)


# (DATA) -----------------------------------------------------------------------
# instead of tf.reshape, as it produces a tensor unknown to .numpy_input_fn()
X1 = np.load(root + 'Data/' + 'P1_X.npy')
train_data = np.reshape(X1[:5, :, :], [-1, 256, 256, 1])
test_data = np.reshape(X1[5:8, :, :], [-1, 256, 256, 1])

Y1 = np.load(root + 'Data/' + 'P1_Y.npy')
train_labels = np.reshape(Y1[:5, :, :], [-1, 256, 256, 1])
test_labels = np.reshape(Y1[5:8, :, :], [-1, 256, 256, 1])


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


DnCNN.train(input_fn=train_input_fn, steps=20)

# (PREDICT) --------------------------------------------------------------------
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x= test_data[0:2,:,:,:],
    y= test_labels[0:2,:,:,:],
    batch_size=1,
    num_epochs=1,
    shuffle=False)

# restoring the model build from estimator: either with
# checkpoints, which is a format dependent on the code that created the model.
# or SavedModel, which is a format independent of the code that created the model.
# in fact model_dir is associated with DnCNN + time stamp of training - it should therefor
# load the correct model

# FIXME predict does not work: label must not be None
predicted = DnCNN.predict(input_fn=test_input_fn) #, checkpoint_path=root+ 'model/' + 'DnCNN_12_17_18_22')
print(list(predicted))

