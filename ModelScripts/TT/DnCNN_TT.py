# Load packages
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
#import gc
import datetime
#import os


train_data = np.load('/scratch2/truhkop/knee1/data/X_train.npy')
train_label = np.load('/scratch2/truhkop/knee1/data/Y_train.npy')



# Definition of the network
def cnn_model_fn (features, labels, mode):
    # Input Layer
    input_layer = features['x']

    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        strides=2,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu,
        name="Conv_1")

    # Convolutional layer #2
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=128,
        strides=[2, 2],
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="Conv_2")

    conv2_bn = tf.layers.batch_normalization(conv2,
                                             name="Conv_2_bn",
                                             center=True,
                                             scale=True,
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Convolutional layer #3
    conv3 = tf.layers.conv2d(
        inputs=conv2_bn,
        filters=256,
        strides=[2, 2],
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="Conv_3")

    conv3_bn = tf.layers.batch_normalization(conv3,
                                             name="Conv_3_bn",
                                             center=True,
                                             scale=True,
                                             training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Deconvolutional layer #1
    deconv1 = tf.layers.conv2d_transpose(
        inputs=conv3_bn,
        filters=256,
        strides=[2, 2],
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="Deconv_1")

    deconv1_bn = tf.layers.batch_normalization(deconv1,
                                               name="deconv_1_bn",
                                               center=True,
                                               scale=True,
                                               training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Deconvolutional layer #2
    deconv2 = tf.layers.conv2d_transpose(
        inputs=deconv1_bn,
        filters=128,
        strides=[2, 2],
        kernel_size=[5, 5],
        activation=tf.nn.relu,
        padding="same",
        name="Deconv_2")

    deconv2_bn = tf.layers.batch_normalization(deconv2,
                                               name="deconv_2_bn",
                                               center=True,
                                               scale=True,
                                               training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Deconvolutional layer #3
    deconv3 = tf.layers.conv2d_transpose(
        inputs=deconv2_bn,
        filters=64,
        strides=[2, 2],
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="Deconv_3")

    deconv3_bn = tf.layers.batch_normalization(deconv3,
                                               name="deconv_3_bn",
                                               center=True,
                                               scale=True,
                                               training=(mode == tf.estimator.ModeKeys.TRAIN))

    # final covolution to get to 3 layers
    conv4 = tf.layers.conv2d(
        inputs=deconv3_bn,
        filters=1,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu,
        name="Conv_4") + input_layer

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=conv4)

    # Calculate Loss (for both Train and EVAL modes)
    loss = tf.losses.absolute_difference(labels=labels, predictions=conv4)
    tf.summary.scalar("Value_Loss_Function", loss)

    for var in tf.trainable_variables():
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()

    # Configure the Training OP (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.mean_absolute_error(
            labels=labels, predictions=conv4)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# ------------------------------------------------------------------------------

runconf = tf.estimator.RunConfig(save_summary_steps=5, log_step_count_steps=10)
save_dir = '/scratch2/truhkop/model/DnCNN_TT' + str(datetime.datetime.now())[0:19].replace(
    "-", "_").replace(" ", "_").replace(
    ":", "_").replace(".", "_")

ImpNet = tf.estimator.Estimator(config=runconf,
                                model_fn=cnn_model_fn, model_dir=save_dir)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_label,
    batch_size=64,
    num_epochs=None,
    shuffle=True)

# run model
ImpNet.train(
    input_fn=train_input_fn,
    steps=10)