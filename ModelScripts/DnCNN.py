import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

# packages needed for MS-SSIM-GL1 loss
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

# optional, must be removed for server
import matplotlib.pyplot as plt
# tf.enable_eager_execution()
# tf.executing_eagerly()


lossflavour = ['MAE', 'MSE', 'SSIM', 'MS-SSIM-GL1'][1]

def DnCNN_model_fn(features, labels, mode):
    """ Beyond a Gaussian Denoiser: Residual learning
        of Deep CNN for Image Denoising.
        Residual learning (originally with X = True + Noisy, Y = noisy)
        :depth: total number of conv-layers in network
        :filters: number of filters for every convolution
        :kernelsize: filter dim: kernelsize by kernelsize
    """

    # workaround, due to tf 1.11.0 rigid argument checking in Estimator
    depth = 4
    filters = 3
    kernelsize = 5


    # (0) Input layer. (batch)-1*256*256*1(channels)
    input_layer = features  # ['X']

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
        inputs=conv_first,
        repetitions=depth - 2,
        layer=tf.contrib.layers.conv2d,
        num_outputs=filters, padding='SAME', stride=1,
        kernel_size=kernelsize,
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                           'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN},
        scope='conv2'
        # passed arguments to conv2d, scope variable to share variables
    )

    # (Debugging skip connections) --------------------------------------------------------------
    # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv2')
    # print([v.name for v in vars])
    # get all operations
    # print(tf.get_default_graph().get_operations())

    # getting the operation.
    # print(tf.get_default_graph().get_operation_by_name('conv2/conv2_2/Conv2D').outputs)
    # print(tf.concat([tf.get_default_graph().get_operation_by_name('conv2/conv2_2/Conv2D').outputs,
    #                 tf.get_default_graph().get_operation_by_name('conv2/conv2_2/Conv2D').outputs], 3))
    # print(tf.stack([tf.get_default_graph().get_operation_by_name('conv2/conv2_2/Conv2D').outputs,
    #                 tf.get_default_graph().get_operation_by_name('conv2/conv2_2/Conv2D').outputs], 1))

    # concatenating along the channel dimension
    # x = tf.constant(-1.0, shape=[1,256, 256,1])
    # tf.concat([x, x], 3)

    # operation
    # print(tf.get_default_graph().get_operation_by_name('conv2/conv2_2/batch_normalization/moving_variance'))

    # tensor
    # print(tf.get_default_graph().get_tensor_by_name('conv2/conv2_2/batch_normalization/moving_variance:0'))
    # x = tf.get_default_graph().get_tensor_by_name('conv2/conv2_2/batch_normalization/moving_variance:0')

    # print(tf.add(x, x))
    # print(tf.concat([x,x], 2))
    # print("#-----------------------------------------------------------------")

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

    prediction = conv_last + input_layer

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prediction
        )

    # For both TRAIN & EVAL:
    # TENSORBOARD
    for var in tf.trainable_variables():
        # write out all variables
        name = var.name
        name = name.replace(':', '_')
        tf.summary.histogram(name, var)
    merged_summary = tf.summary.merge_all()

    # (losses) -----------------------------------------------------------------
    def l1(prediction, labels):
        return tf.losses.absolute_difference(
            labels=labels,
            predictions=prediction)

    def mse(prediction, labels):
        return tf.losses.mean_squared_error(
            labels=labels,
            predictions=prediction)

     # throws an error
    def ssim(prediction, labels):
        # ssim returns a tensor containing ssim value for each image in batch: reduce!
        return 1 - tf.reduce_mean(
            tf.image.ssim(
                prediction,
                labels,
                max_val=1))

    def loss_ssim_multiscale_gl1(prediction, label, alpha = 0.84):
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

        # BATCHNROM 'memoize' : look at tf.contrib.layers.batch_norm doc.
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

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
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

DnCNN = tf.estimator.Estimator(
    model_fn=DnCNN_model_fn,
    model_dir=root + 'model/DnCNN_' +
              "{}_{}_{}_{}_{}".format(lossflavour, d.month, d.day, d.hour, d.minute),
    config=tf.estimator.RunConfig(
        save_summary_steps=2,
        log_step_count_steps=2)
)
print("generated {}_{}_{}_{}_{} Estimator".format(lossflavour, d.month, d.day, d.hour,
                                                     d.minute))


# (DATA) -----------------------------------------------------------------------
# instead of tf.reshape, as it produces a tensor unknown to .numpy_input_fn()

def np_read_patients(root, patients=range(1, 4)):
    '''Read and np.concatenate the patients arrays for noise and true image.'''
    def helper(string):
        return np.concatenate(
            tuple(np.load(root + arr) for arr in
                  list('P{}_{}.npy'.format(i, string) for i in patients)),
            axis=0)

    return helper('X'), helper('Y')


def subset_arr(X, Y, batchind=range(5)):
    '''intended to easily subset data into train and test.
    :return: a Tuple of  two arrays resulted from the subset: X and Y in this order.
    The arrays are reshaped to [batchsize, hight, width, channels]'''
    return tuple(np.reshape(z[batchind, :, :], [-1, 256, 256, 1]) for z in [X, Y])


X, Y = np_read_patients(
    root='C:/Users/timru/Documents/CODE/deepMRI1/data/',
    patients=range(1, 2))

train_data, train_labels = subset_arr(X, Y, batchind=range(5))
test_data, test_labels = subset_arr(X, Y, batchind=range(5, 8))

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

DnCNN.train(input_fn=train_input_fn, steps=2)

# (PREDICTING) -----------------------------------------------------------------
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=test_data,
    y=test_labels,
    shuffle=False)

predicted = DnCNN.predict(input_fn=test_input_fn)  # , checkpoint_path=root +
# 'model/' + "DnCNN_{}_{}_{}_{}".format(d.month, d.day, d.hour, d.minute))
pred = list(predicted)



# (plot the predicted) ---------------------------------------------------------
def inspect_images(noisy, pred, true, index, scaling=256):
    """
    :param noisy: Array of noisy input images
    :param pred: Array of predicted images
    :param true: Array of ground truth images
    :param index: Images index in all of the above, which shall be plotted in
    parallel
    :param scaling: factor, with which the images are scaled (input
    originally was downscaled by scaling**-1
    """

    plt.figure(1)
    plt.subplot(131)
    plt.title('Noisy Input')
    plt.imshow(np.reshape(noisy[index], (256, 256)) * scaling, cmap='gray')

    plt.subplot(132)
    plt.title('Prediction')
    plt.imshow(np.reshape(pred[index], (256, 256)) * scaling, cmap='gray')

    plt.subplot(133)
    plt.title('Truth')
    plt.imshow(np.reshape(true[index], (256, 256)) * scaling, cmap='gray')

    plt.show()


inspect_images(noisy=test_data, pred=pred, true=test_labels, index=0)
inspect_images(noisy=test_data, pred=pred, true=test_labels, index=1)

# plt any image of the original tensor
# plt.imshow(np.reshape(X[3600], (256, 256)) * 256, cmap='gray')
# plt.show()




# (EVALUATION) -----------------------------------------------------------------
valid_data = test_data
valid_labels = test_labels

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=valid_data,
    y=valid_labels,
    batch_size=1,
    num_epochs=1,
    shuffle=False)

DnCNN.evaluate(input_fn=test_input_fn)
print('Evaluated REDnet_L1_{}_{}_{}_{} Estimator'.format(d.month, d.day, d.hour, d.minute))




# restore an estimator from last checkpoints ----------------------------------
# restoring the model build from estimator: either with
# checkpoints, which is a format dependent on the code that created the model.
# or SavedModel, which is a format independent of the code that created the
# model. In fact model_dir is associated with DnCNN + time stamp of
# training - it should therefor
# load the correct model

restoreDnCNN = tf.estimator.Estimator(
    model_fn=DnCNN_model_fn,
    model_dir=root + 'model/' + 'DnCNN_model_restored',
    warm_start_from='C:/Users/timru/Documents/CODE'
                    '/deepMRI1/model/DnCNN_12_26_17_21')

predictfromrestored = list(restoreDnCNN.predict(test_input_fn))





# NOTE skimage not yet available on server
sum_mae = 0
sum_mse = 0
sum_ssim = 0
for im_num in range(0, test_labels.shape[0]):
    prediction = next(predictfromrestored)
    true_image = test_labels[im_num,:,:,:]
    sum_mae += np.mean(np.abs(prediction - true_image))
    sum_mse += np.mean(np.power((prediction - true_image), 2))
    # sum_ssim += skimage.measure.compare_ssim(prediction[:,:,0],  true_image[:,:,0])
    if(im_num % 500 == 0):
        print("Current mae is " + str(sum_mae) + " and mse is " + str(sum_mse) + " and ssim is " + str(sum_ssim) + " at picture " + str(im_num))
# 	sci.imsave(root + 'predictedImg/DnCNN_MSE_2_1_23_no{}.jpg'.format(im_num),
#                  np.reshape(prediction*255, (256, 256,1)))

print('MAE', sum_mae)
print('MSE', sum_mse)
print('SSIM', sum_ssim)

