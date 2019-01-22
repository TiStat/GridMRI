import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def REDnet_model_fn (features, labels, mode):
    """Image Restoration Using Convolutional Auto-encoders with Symmetric Skip-Connections
    Conv & Deconv_k + ReLu: F(X) = max(0, W_k * X + B_k)
    Skip connection: F(X_1, X_2) = max(0, X_1 + X_2)
    Total of 20 or 30 layers

    :reps: number of repeated application of convoltuions/deconvolutions in one level
    :filters: list of numbers of out-filters for all convs/deconvs repetition.
    the len of filters is the number of levels.
    :kernelsize: size of each filter.
    """

    # (new downward path) working on its own -----------------------------------
    filters = [2, 4]
    # filters = [64,128,256,512,1024]
    kernelsize = 5
    reps = 2

    def leveld (inputs, filters, scope, kwargs):
        print(inputs)
        return tf.contrib.layers.repeat(
            inputs=inputs,
            num_outputs=filters,
            scope=scope,
            **kwargs)

    d = tf.contrib.layers.stack(inputs=features,
                                layer=leveld,
                                stack_args=filters,
                                kwargs={'repetitions': reps,
                                        'layer': tf.contrib.layers.conv2d,
                                        'kernel_size': 5,
                                        'padding': 'SAME',
                                        'stride': 1,
                                        'activation_fn': tf.nn.relu}
                                )

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print([v.name for v in vars])
    print(d)

    # (upward patch) ----------------------------------------------------------
    def levelu (inputs, concatin, filters, scope):
        '''anzahl der elemente der filter gibt vor, wieviele ebenen entstehen,
        reps entscheided, wieviele conv/deconvs in einer ebene sind'''
        print(concatin)
        print(inputs)

        # skip connection: extracting the last tensor of the stack/repeat call.
        inputs = tf.concat([inputs, tf.reshape(concatin, tf.shape(concatin)[1:])], axis=3, name='insert')

        # deconvolution
        return tf.contrib.layers.repeat(
            inputs=inputs,
            repetitions=reps,
            layer=tf.contrib.layers.conv2d_transpose,
            num_outputs=filters,
            padding='SAME',
            stride=1,
            kernel_size=[kernelsize, kernelsize],
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.layers.batch_normalization,
            normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                               'trainable': False,
                               'training': mode == tf.estimator.ModeKeys.TRAIN},
            scope=scope
        )

    l = list(range(len(filters)))
    for i, v in enumerate(filters[::-1]):
        # print((i, v, l[-(i+1)]))
        d = levelu(
            inputs=d,
            concatin=tf.get_default_graph().get_operation_by_name('Stack/leveld_' + str(i + 1) + '/leveld_' + str(
                i + 1) + '_' + str(reps) + '/Conv2D').outputs,  # 3 durch + str(reps) + ersetzen??
            filters=v,
            scope='level' + str(l[-(i + 1)])
        )

    print(d)

    # (last layer) -------------------------------------------------------------
    residual = tf.contrib.layers.conv2d_transpose(
        inputs=d,
        kernel_size=kernelsize,
        stride=1,
        num_outputs=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                           'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN}
    )

    # (i) Conv + ReLu. Filter: 32x5x5 or even 64
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

    # use conv1 to skip

    # (ii) Deconv + (skip connection: elementwise sum) + ReLu
    # tf.contrib.layers.conv2d_transpose(

    # PREDICTION
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=residual + features
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
    # (iii).1 Loss (Mean-Squared-Error)
    # (iii).2 Optimization (ADAM, base learning rate: 10^-4)
    #         for details on momentum vectors, update rule, see recommended values p.6
    # (L1 Version)
    loss = tf.losses.absolute_difference(
        labels=labels,
        predictions=residual + features)
    tf.summary.scalar("Value_Loss_Function", loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
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
                "accuracy": tf.metrics.mean_absolute_error(
                    labels=labels,
                    predictions=residual + features)
            }
        )

    # Adam details:
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/


# (ESTIMATOR) ------------------------------------------------------------------
# root = '/home/cloud/' # for jupyter
root = 'C:/Users/timru/Documents/CODE/deepMRI1/'
d = datetime.datetime.now()

REDnet = tf.estimator.Estimator(
    model_fn=REDnet_model_fn,
    model_dir=root + 'model/' +
              "REDnet_{}_{}_{}_{}".format(d.month, d.day, d.hour, d.minute),
    config=tf.estimator.RunConfig(save_summary_steps=2,
                                  log_step_count_steps=2)
)

print("generated REDnet_{}_{}_{}_{} Estimator".format(d.month, d.day, d.hour, d.minute))


# (DATA) -----------------------------------------------------------------------
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


X, Y = np_read_patients(root='C:/Users/timru/Documents/CODE/deepMRI1/data/',
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

REDnet.train(input_fn=train_input_fn, steps=20)

# (PREDICTION) -----------------------------------------------------------------
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=test_data,
    y=test_labels,
    batch=1,
    num_epochs=None,
    shuffle=True)

predicted = REDnet.predict(input_fn=test_input_fn)  # , checkpoint_path=root +
# 'model/' + "DnCNN_{}_{}_{}_{}".format(d.month, d.day, d.hour, d.minute))
pred = list(predicted)


# (plot the predicted) ---------------------------------------------------------
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
