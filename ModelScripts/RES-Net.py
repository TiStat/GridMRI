# import matplotlib.pyplot as plt
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss

lossflavour = ['MAE', 'MSE', 'SSIM', 'MS-SSIM-GL1'][1]
def RESnet_model_fn (features, labels, mode):
    """Autoencoder with symmetric skip connections
    :filters: number of filters applied to in the downward path (and reverse in upward path)
    minimal length of list is two
    :reps: parameter of how many repetitions before skipping. minimal value is 1
    """

    filters = [2, 4]
    kernelsize = 10
    reps = 2

    def leveld (inputs, filters, scope):
        d = tf.contrib.layers.repeat(
            inputs=inputs,
            num_outputs=filters,
            scope=scope,
            repetitions=reps,
            layer=tf.contrib.layers.conv2d,
            kernel_size=kernelsize,
            padding='valid',
            stride=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
                'trainable': True,
                'training': mode == tf.estimator.ModeKeys.TRAIN})
        # print(d)
        return d

    def levelu (inputs, concatin, filters, scope):
        # skip connection: extracting the last tensor of the stack/repeat call.
        # print(inputs, 'concatinating to ', concatin)
        inputs = tf.concat([inputs, tf.reshape(concatin, tf.shape(concatin)[1:])], axis=3, name='insert')

        # deconvolution
        d = tf.contrib.layers.repeat(
            inputs=inputs,
            repetitions=reps,
            layer=tf.contrib.layers.conv2d_transpose,
            num_outputs=filters,
            padding='valid',
            stride=1,
            kernel_size=kernelsize,
            activation_fn=tf.nn.relu,
            scope=scope,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
                'trainable': True,
                'training': mode == tf.estimator.ModeKeys.TRAIN}
        )
        # print(d)
        return d

    # (downward path) ----------------------------------------------------------
    # print(features)
    d = tf.contrib.layers.stack(
        inputs=features,
        layer=leveld,
        stack_args=filters[:-1],
        scope='stackdown'
    )

    # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # print([v.name for v in vars])
    # print(d)

    # (lowest) -----------------------------------------------------------------

    d = tf.contrib.layers.conv2d(
        inputs=d,
        num_outputs=filters[-1],
        padding='valid',
        stride=1,
        kernel_size=kernelsize,
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={
            'momentum': 0.99,
            'epsilon': 0.001,
            'trainable': True,
            'training': mode == tf.estimator.ModeKeys.TRAIN}
    )
    # print(d)
    d = tf.contrib.layers.conv2d_transpose(
        inputs=d,
        num_outputs=filters[-1],
        padding='valid',
        stride=1,
        kernel_size=kernelsize,
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={
            'momentum': 0.99,
            'epsilon': 0.001,
            'trainable': True,
            'training': mode == tf.estimator.ModeKeys.TRAIN}
    )


    # print(d)

    # (upward patch) -----------------------------------------------------------
    for i, v in zip(list(range(1, len(filters)))[::-1], filters[-2::-1]):
        d = levelu(
            inputs=d,
            concatin=tf.get_default_graph().get_operation_by_name(
                'Stack/leveld_{}/leveld_{}_{}/Conv2D'.format(str(i), str(i), str(reps))).outputs,
            filters=v,
            scope='level_' + str(i)
        )

    # (last layer) -------------------------------------------------------------
    residual = tf.contrib.layers.conv2d_transpose(
        inputs=d,
        kernel_size=kernelsize,
        stride=1,
        num_outputs=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={
            'momentum': 0.99,
            'epsilon': 0.001,
            'trainable': False,
            'training': mode == tf.estimator.ModeKeys.TRAIN},
    )
    # print(residual)
    prediction = residual + features

    # convdown = tf.contrib.layers.stack(inputs=features, layer=repconv, stack_args=[item for item in filters for i in
    # range(reps)])

    # variable scope access like this during session?
    # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv2')
    # print([v.name for v in vars])
    # get all operations
    # print(tf.get_default_graph().get_operations())

    # getting the operation.
    # based on 'scope/variable/operation'.outputtensor
    # print(tf.get_default_graph().get_operation_by_name('conv2/conv2_2/Conv2D').outputs)

    # PREDICTION
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prediction  # final skip connection
        )

    # TENSORBOARD
    # For both TRAIN & EVAL:
    for var in tf.trainable_variables():
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()

    # TRAINING
    # (losses) -----------------------------------------------------------------
    def l1 (prediction, labels):
        return tf.losses.absolute_difference(
            labels=labels,
            predictions=prediction)

    def mse (prediction, labels):
        return tf.losses.mean_squared_error(
            labels=labels,
            predictions=prediction)

    # throws an error
    def ssim (prediction, labels):
        # ssim returns a tensor containing ssim value for each image in batch: reduce!
        return 1 - tf.reduce_mean(
            tf.image.ssim(
                prediction,
                labels,
                max_val=1))

    def loss_ssim_multiscale_gl1 (prediction, label, alpha=0.84):
        ''' Loss function, calculating alpha * Loss_msssim + (1-alpha) gaussiankernel * L1_loss
        according to 'Loss Functions for Image Restoration with Neural Networks' [Zhao]
        :alpha: default value accoording to paper'''

        # stride according to MS-SSIM source
        kernel_on_l1 = tf.nn.conv2d(
            input=tf.subtract(label, prediction),
            filter=gaussiankernel,
            strides=[1, 1, 1, 1],
            padding='VALID')

        # total no. of pixels: number of patches * number of pixels per patch
        img_patch_norm = tf.to_float(kernel_on_l1.shape[1] * filter_size ** 2)
        gl1 = tf.reduce_sum(kernel_on_l1) / img_patch_norm

        # ssim_multiscale already calculates the dyalidic pyramid (with as replacment avg.pooling)
        msssim = tf.reduce_sum(
            tf.image.ssim_multiscale(
                img1=label,
                img2=prediction,
                max_val=1)
        )
        return alpha * (1 - msssim) + (1 - alpha) * gl1

    # Discrete Gaussian Kernel (required only in MS-SSIM-GL1 case)
    # not in MS-SSIM-GL1 function, as it is executed only once
    # values according to MS-SSIM source code
    filter_size = constant_op.constant(11, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(1.5, dtype=features.dtype)
    gaussiankernel = _fspecial_gauss(
        size=filter_size,
        sigma=filter_sigma
    )

    # for TRAIN & EVAL
    loss = {
        'MAE': l1,
        'MSE': mse,
        'SSIM': ssim,
        'MS-SSIM-GL1': loss_ssim_multiscale_gl1
    }[lossflavour](prediction, labels)
    tf.summary.scalar("Value_Loss_Function", loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.histogram('Summary_final_layer', prediction)
        tf.summary.histogram('Summary_labels', labels)
        tf.summary.image('Input_Image', features)
        tf.summary.image('Output_Image', prediction)
        tf.summary.image('True_Image', labels)

        # BATCHNROM 'memoize'
        # look at tf.contrib.layers.batch_norm doc.
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
                "Mean absolute error (MAE)": tf.metrics.mean_absolute_error(
                    labels=labels,
                    predictions=prediction),
                'Mean squared error (MSE)': tf.metrics.mean_squared_error(
                    labels=labels,
                    predictions=prediction)
                # ,
                # 'Structural Similarity Index (SSIM)': tf.image.ssim(
                #     img1=prediction,
                #     img2=labels,
                #     max_val=1) # needs custom eval_metric_opc function, returning (metric value, update_ops)
            }
        )

    # Adam details:
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/


# (ESTIMATOR) ------------------------------------------------------------------
# root = '/home/cloud/' # for jupyter
root = 'C:/Users/timru/Documents/CODE/deepMRI1/'
d = datetime.datetime.now()

RESnet = tf.estimator.Estimator(
    model_fn=RESnet_model_fn,
    model_dir=root + 'model/RESnet_' +
              "{}_{}_{}_{}_{}".format(lossflavour, d.month, d.day, d.hour, d.minute),
    config=tf.estimator.RunConfig(
        save_summary_steps=2,
        log_step_count_steps=2)
)
print("generated RESnet_{} {}_{}_{}_{} Estimator".format(lossflavour, d.month, d.day, d.hour,
                                                      d.minute))


def np_read_patients (root, patients=range(1, 4)):
    '''Read and np.concatenate the patients arrays for noise and true image.'''

    def helper (string):
        return np.concatenate(
            tuple(np.load(root + arr) for arr in
                  list('P{}_{}.npy'.format(i, string) for i in patients)),
            axis=0)

    return helper('X'), helper('Y')


def subset_arr (X, Y, batchind=range(5)):
    '''intended to easily subset data into train and test.
    :return: a Tuple of  two arrays resulted from the subset: X and Y in this order.
    The arrays are reshaped to [batchsize, height, width, channels]'''
    return tuple(np.reshape(z[batchind, :, :], [-1, 256, 256, 1]) for z in [X, Y])


X, Y = np_read_patients(root='C:/Users/timru/Documents/CODE/deepMRI1/data/',
                        patients=range(1, 2))

train_data, train_label = subset_arr(X, Y, batchind=range(5))
test_data, test_label = subset_arr(X, Y, batchind=range(5, 8))

# (TRAINING) -------------------------------------------------------------------
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_data,
    y=train_label,
    batch_size=2,
    num_epochs=None,
    shuffle=True)

RESnet.train(input_fn=train_input_fn, steps=2)

# (TESTING) --------------------------------------------------------------------
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=test_data[0:2],
    y=test_label[0:2],
    batch_size=1,
    num_epochs=None,
    shuffle=True)

predicted = RESnet.predict(input_fn=test_input_fn)  # , checkpoint_path=root +
# 'model/' + "DnCNN_{}_{}_{}_{}".format(d.month, d.day, d.hour, d.minute))
pred = list(predicted)

