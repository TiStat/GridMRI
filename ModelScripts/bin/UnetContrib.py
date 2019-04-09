import datetime
import numpy as np
import tensorflow as tf

from tensorflow.python.ops.image_ops_impl import _fspecial_gauss
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

#tf.enable_eager_execution()
#tf.executing_eagerly()

training = True
predict = False
platform = ['hpc', 'cloud', 'home'][2]
root = {'hpc':'/scratch2/truhkop/',
        'cloud':'/home/cloud/',
        'home':'C:/Users/timru/Documents/CODE/deepMRI1/'} [platform]
lossflavour = ['MAE', 'MSE', 'SSIM','MS-SSIM', 'MS-SSIM-GL1'][1]
filters = [1,2,3,4,5] # always uneven length
kernel = [5,7,10][0]
reps = [1,2,3,4,5]
dncnn_skip = True # a dncnn-like skip connection throughout the entire network
batch = [2,6,10][0]
trainsteps = 100

len(filters) % 2 == 1
len(filters) == len(reps)

def Unet_model_fn (features, labels, mode):

    # nodes of tf.graph, stored in list, that extends with each operation
    # therefore direct path: input=nodes[-1]
    nodes = list()
    nodes.append(features)
    depth = len(filters)

    # downward path
    for i in range(depth//2):
        nodes.append(tf.contrib.layers.repeat(
            inputs=nodes[-1],
            num_outputs=filters[i],
            repetitions=reps[i],
            layer=tf.contrib.layers.conv2d,
            kernel_size=kernel,
            padding='SAME',
            stride=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
                'trainable': False,
                'training': mode == tf.estimator.ModeKeys.TRAIN}
        ))

        nodes.append(tf.layers.max_pooling2d(
            inputs=nodes[-1],
            pool_size=2,
            strides=1,
            padding='VALID',
        ))

    nodes.append(tf.contrib.layers.repeat(
            inputs=nodes[-1],
            num_outputs=filters[depth//2],
            repetitions=reps[depth//2],
            layer=tf.contrib.layers.conv2d,
            kernel_size=kernel,
            padding='SAME',
            stride=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
                'trainable': False,
                'training': mode == tf.estimator.ModeKeys.TRAIN}
    ))

    # upward path
    for i in range(depth-2, 0,-2): # range((depth//2)+1, 0 , -2):
        nodes.append(tf.contrib.layers.conv2d_transpose(
            inputs=nodes[-1], # always takes previous conv2d
            kernel_size=2,
            stride=1,
            num_outputs=filters[-i],
            padding='VALID',
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
                'trainable': False,
                'training': mode == tf.estimator.ModeKeys.TRAIN},
        ))

        skip = tf.concat([nodes[i], nodes[-1]], axis = 3)

        nodes.append(tf.contrib.layers.repeat(
            inputs=skip,
            num_outputs=filters[i],
            repetitions=reps[i],
            layer=tf.contrib.layers.conv2d,
            kernel_size=kernel,
            padding='SAME',
            stride=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
                'trainable': False,
                'training': mode == tf.estimator.ModeKeys.TRAIN}
        ))

    prediction = tf.contrib.layers.conv2d(
        inputs = nodes[-1],
        num_outputs=1,
        kernel_size=kernel,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu
    )

    if dncnn_skip:
        prediction = prediction +  features

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=prediction)

    # TENSORBOARD
    # For both TRAIN & EVAL:
    for var in tf.trainable_variables():
        # write out all variables
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

    def ssim (prediction, labels):
        # ssim returns a tensor containing ssim value for each image in batch: reduce!
        return 1 - tf.reduce_mean(
            tf.image.ssim(
                prediction,
                labels,
                max_val=1))

    def ssim_multiscale(prediction, label):
        return 1- tf.reduce_mean(
            tf.image.ssim_multiscale(
                    img1=label,
                    img2=prediction,
                    max_val=1)
            )

    def ssim_multiscale_gl1 (prediction, label, alpha=0.84):
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
        msssim = tf.reduce_mean(
            tf.image.ssim_multiscale(
                img1=label,
                img2=prediction,
                max_val=1)
        )
        return alpha * (1 - msssim) + (1 - alpha) * gl1

    if lossflavour == 'MS-SSIM-GL1':
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
        'MS-SSIM': ssim_multiscale,
        'MS-SSIM-GL1': ssim_multiscale_gl1
    }[lossflavour](prediction, labels)
    tf.summary.scalar("Value_Loss_Function", loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.histogram('Summary_final_layer', prediction)
        tf.summary.histogram('Summary_labels', labels)
        tf.summary.image('Input_Image', features)
        tf.summary.image('Output_Image', prediction)
        tf.summary.image('True_Image', labels)
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


# (ESTIMATOR) ------------------------------------------------------------------
d = datetime.datetime.now()

Unet = tf.estimator.Estimator(
    model_fn=Unet_model_fn,
    model_dir=root + 'model/Unet_' +
              "{}_{}_{}_{}_{}".format(lossflavour, d.month, d.day, d.hour, d.minute),
    config=tf.estimator.RunConfig(
        save_summary_steps=2,
        log_step_count_steps=2)
)

print("generated Unet_{}_{}_{}_{}_{} Estimator".format(lossflavour, d.month, d.day, d.hour, d.minute))


# (DATA) -----------------------------------------------------------------------
if platform == 'hpc':
    if training:
        train_data = np.load('/scratch2/truhkop/knee1/data/X_train.npy')
        train_labels = np.load('/scratch2/truhkop/knee1/data/Y_train.npy')

    if predict:
        test_data= np.load('/scratch2/truhkop/knee1/data/X_test.npy')
        test_labels=  np.load('/scratch2/truhkop/knee1/data/Y_test.npy')

elif platform == 'cloud':
    if training:
        train_data = np.load(root +'/data/X_test_subset.npy')
        train_labels = np.load(root + '/data/Y_test_subset.npy')

elif platform == 'home':
    # instead of tf.reshape, as it produces a tensor unknown to .numpy_input_fn()

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
        The arrays are reshaped to [batchsize, hight, width, channels]'''
        return tuple(np.reshape(z[batchind, :, :], [-1, 256, 256, 1]) for z in [X, Y])

    X, Y = np_read_patients(
        root='C:/Users/timru/Documents/CODE/deepMRI1/data/',
        patients=range(1, 2))

    if training:
        train_data, train_labels = subset_arr(X, Y, batchind=range(20))

    if predict:
        test_data, test_labels = subset_arr(X, Y, batchind=range(5, 8))


# (TRAINING) -------------------------------------------------------------------
if training == True:
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=batch,
        num_epochs=None,
        shuffle=True)

    Unet.train(input_fn=train_input_fn, steps=trainsteps)