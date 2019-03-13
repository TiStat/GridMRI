import datetime
import numpy as np
import tensorflow as tf
import RESNet

# packages needed for MS-SSIM-GL1 loss
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

# optional, must be removed for server
# tf.enable_eager_execution()
# tf.executing_eagerly()



training = True
predict = False
platform = ['hpc', 'cloud', 'home'][2]
root = {'hpc':'/scratch2/truhkop/',
        'cloud':'/home/cloud/',
        'home':'C:/Users/timru/Documents/CODE/deepMRI1/'} [platform]
lossflavour = ['MAE', 'MSE', 'SSIM','MS-SSIM', 'MS-SSIM-GL1'][1]
filters = [2, 4]
kernelsize = 5
reps = 3


def REDnet_model_fn (features, labels, mode):
    """Image Restoration Using Convolutional Auto-encoders with Symmetric Skip-Connections
    :reps: number of repeated application of convoltuions/deconvolutions in one level
    :filters: list of numbers of out-filters for all convs/deconvs repetition.
    the len of filters is the number of levels.
    :kernelsize: size of each filter.
    """


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
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
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
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
                'trainable': False,
                'training': mode == tf.estimator.ModeKeys.TRAIN}
        )
        # print(d , concatin)

        # skip connection: extracting the last tensor of the stack/repeat call.
        d = tf.concat([d, tf.reshape(concatin, tf.shape(concatin)[1:])],
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
            normalizer_params={
                'momentum': 0.99,
                'epsilon': 0.001,
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
        normalizer_params={
            'momentum': 0.99,
            'epsilon': 0.001,
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
    residual = tf.contrib.layers.conv2d( # (1*1*no of channels within) aggregating over all filters
        inputs=d,
        kernel_size=1,
        stride=1,
        num_outputs=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
    )

    prediction = residual + features

    # variable scope access like this during session?
    # vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # for v in vars:
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
            predictions=prediction
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
        'MS-SSIM-GL1': loss_ssim_multiscale_gl1
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

REDnet = tf.estimator.Estimator(
    model_fn=REDnet_model_fn,
    model_dir=root + 'model/REDnet_' +
              "{}_{}_{}_{}_{}".format(lossflavour, d.month, d.day, d.hour, d.minute),
    config=tf.estimator.RunConfig(
        save_summary_steps=2,
        log_step_count_steps=2)
)

print("generated REDnet_{}_{}_{}_{} Estimator".format(d.month, d.day, d.hour, d.minute))


# (DATA) -----------------------------------------------------------------------
if platform == 'hpc':
    if training:
        train_data = np.load('/scratch2/truhkop/knee1/data/X_train.npy')
        train_labels = np.load('/scratch2/truhkop/knee1/data/Y_train.npy')

    if predict:
        test_data= np.load('/scratch2/truhkop/knee1/data/X_test.npy')
        test_labels=  np.load('/scratch2/truhkop/knee1/data/Y_test.npy')

elif platform == 'cloud':
    pass

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
# learning with mini batch (128 images), 50 epochs
# rewrite with tf.placeholder, session.run
# https://stackoverflow.com/questions/49743838/predict-single-image-after-training-model-in-tensorflow
if training == True:
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=2,
        num_epochs=None,
        shuffle=True)

    REDnet.train(input_fn=train_input_fn, steps=10)

# (PREDICTION) -----------------------------------------------------------------
if predict == True:
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_data,
        y=test_labels,
        batch_size=1,
        num_epochs=None,
        shuffle=True)

    restoreREDnet = tf.estimator.Estimator(
        model_fn=REDnet_model_fn,
        model_dir=root + 'model/' + 'REDnet_model_restored',
        warm_start_from='C:/Users/timru/Documents/CODE'
                        '/deepMRI1/model/REDnet_MSE_3_3_9_11')

    predictfromrestored = list(restoreREDnet.predict(test_input_fn))
    pred = list(predictfromrestored)


    # predicted = REDnet.predict(input_fn=test_input_fn)  # , checkpoint_path=root +
    # # 'model/' + "DnCNN_{}_{}_{}_{}".format(d.month, d.day, d.hour, d.minute))
    # pred = list(predicted)

    # (plot the predicted) ---------------------------------------------------------
    import matplotlib.pyplot as plt
    def inspect_images (noisy, pred, true, index, scaling=256):
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
