import datetime
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def RESnet_model_fn(features, labels, mode):
    """Autoencoder with symmetric skip connections
    :filters: number of filters applied to in the downward path (and reverse in upward path)
    minimal length of list is two
    :reps: parameter of how many repetitions before skipping. minimal value is 1
    """

    # filters = [64, 128, 256, 512, 1024]
    filters = [2, 4]
    kernelsize = 10
    reps = 1

    def leveld(inputs, filters, scope, kwargs):
        d = tf.contrib.layers.repeat(
            inputs=inputs,
            num_outputs=filters,
            scope=scope,
            **kwargs)
        print(d)
        return d

    def levelu(inputs, concatin, filters, scope):
        # skip connection: extracting the last tensor of the stack/repeat call.
        print(inputs, 'concatinating to ', concatin)
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
            normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                               'trainable': False,
                               'training': mode == tf.estimator.ModeKeys.TRAIN}
            # passed arguments to conv2d, scope variable to share variables
        )
        print(d)
        return d

    # (downward path) ----------------------------------------------------------
    print(features)
    d = tf.contrib.layers.stack(
        inputs=features,
        layer=leveld,
        stack_args=filters[:-1],
        kwargs={'repetitions': reps,
                'layer': tf.contrib.layers.conv2d,
                'kernel_size': kernelsize,
                'padding': 'valid',
                'stride': 1,
                'activation_fn': tf.nn.relu,
                'normalizer_fn': tf.layers.batch_normalization,
                'normalizer_params': {'momentum': 0.99,
                                      'epsilon': 0.001,
                                      'trainable': False,
                                      'training': mode == tf.estimator.ModeKeys.TRAIN}
    }
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
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                           'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN}
    )
    print(d)
    d = tf.contrib.layers.conv2d_transpose(
        inputs=d,
        num_outputs=filters[-1],
        padding='valid',
        stride=1,
        kernel_size=kernelsize,
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                           'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN}
    )

    print(d)
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
        normalizer_params={'momentum': 0.99, 'epsilon': 0.001,
                           'trainable': False,
                           'training': mode == tf.estimator.ModeKeys.TRAIN}
    )

    print(residual)

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
            predictions=residual + features  # final skip connection
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

RESnet = tf.estimator.Estimator(
    model_fn=RESnet_model_fn,
    model_dir=root + 'model/' +
              "RESnet_{}_{}_{}_{}".format(d.month, d.day, d.hour, d.minute),
    config=tf.estimator.RunConfig(save_summary_steps=2,
                                  log_step_count_steps=2)
)

print("generated RESnet_{}_{}_{}_{} Estimator".format(d.month, d.day, d.hour,
                                                      d.minute))


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

RESnet.train(input_fn=train_input_fn, steps=20)

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
