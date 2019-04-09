import tensorflow as  tf

# MS-SSIM-GL1 related
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss

# Debugging tf.graph:
#tf.enable_eager_execution()
#tf.executing_eagerly()


# (Layers) -----------------------------------------------------------------
def convrep (mode, rep, inp, filt, ker):
    x = inp
    for i in range(rep):
        x = tf.layers.conv2d(
            inputs=x,
            filters=filt,
            kernel_size=ker,
            padding="same",
            activation=None)
        # print(x.shape, 'conv')

        x = tf.layers.batch_normalization(
            x,
            axis=-1,
            training=(mode == tf.estimator.ModeKeys.TRAIN))

        x = tf.nn.relu(x)

    return x


def convtransp (mode, tensor_in, filt, ker):
    x = tf.layers.conv2d_transpose(
        tensor_in,
        filters=filt,
        kernel_size=2,
        strides=2,
        padding='same',
        activation=None)

    x = tf.layers.batch_normalization(
        x,
        axis=-1,
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    x = tf.nn.relu(x)
    # print(x.shape, 'convtransp')
    return x

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
    # ssim returns a tensor containing ssim value
    # for each image in batch: reduce!
    # note, that SSIM is to be maximized
    return -1 * (1 - tf.reduce_mean(
        tf.image.ssim(
            prediction,
            labels,
            max_val=1)))


def ssim_multiscale (prediction, label):
    return -1 * (tf.reduce_mean(
        tf.image.ssim_multiscale(
            img1=label,
            img2=prediction,
            max_val=1 #,
            #power_factors=(0.0448, 0.2856, 0.3001, 0.2363)
        )
    ))


def ssim_multiscale_gl1 (prediction, label, alpha=0.84):
    ''' Loss function, calculating alpha * Loss_msssim + (1-alpha) gaussiankernel * L1_loss
    according to 'Loss Functions for Image Restoration with Neural Networks' [Zhao]
    :alpha: default value accoording to paper'''

    # Discrete Gaussian Kernel (required only in MS-SSIM-GL1 case)
    # TODO decorator function for single instance of kernel, also features is only in envir not arg!
    filter_size = constant_op.constant(11, dtype=dtypes.int32)
    filter_sigma = constant_op.constant(1.5, dtype=features.dtype)
    gaussiankernel = _fspecial_gauss(
        size=filter_size,
        sigma=filter_sigma
    )

    # stride according to MS-SSIM source
    kernel_on_l1 = tf.nn.conv2d(
        input=tf.subtract(label, prediction),
        filter=gaussiankernel,
        strides=[1, 1, 1, 1],
        padding='VALID')

    # total no. of pixels: number of patches * number of pixels per patch
    img_patch_norm = tf.to_float(kernel_on_l1.shape[1] * filter_size ** 2)
    gl1 = tf.reduce_sum(kernel_on_l1) / img_patch_norm

    msssim = -1 * (tf.reduce_mean(
        tf.image.ssim_multiscale(
            img1=label,
            img2=prediction,
            max_val=1  # ,
            # power_factors=(0.0448, 0.2856, 0.3001, 0.2363))
        )))
    return alpha * (1 - msssim) + (1 - alpha) * gl1

# (model_fn) -----------------------------------------------------------------
def Unet_model_fn (features, labels, mode, params):
    '''
    :params kernel: kernelsize, skalar parameter for conv kernel,
    :params filters: list, specifying for all convs in a conv_rep block, how many
    output filters (kernel) are to beapplied
    :params lossflavour: string, specifying which loss function is to be used.
    MS-SSIM, MS-SSIM-GL1 are not tested eagerly. Can be any of:
    ['MSE', 'MAE', 'SSIM', 'MS-SSIM','MS-SSIM-GL1']
    :params batch: size of the batch, applied to all conv layers
    :params reps: list, giving the number of stacked convs in the respective conv_rep block
    :params dncnn_skip: boolean, optional skip connectionen from input (x)
    to prediction. (p = x + netoutput)
    :params sampling: boolean  whether or whether not to up /downsample
    the image with pool/deconv
    :params Uskip: boolean, whether or whether not to use the unet skipping scheme
    :params trainsteps: integer, number of training steps for all model configurations
        to obtain a DnCNN like architechture set
        dncnn_skip = True
        samplind = False
        Uskip = False'''

    kernel = params['kernel']
    filters = params['filters']
    lossflavour = params['lossflavour']
    reps = params['reps']
    dncnn_skip = params['dncnn_skip']
    sampling = params['sampling']
    Uskip = params['Uskip']

    # Developer note on skipping scheme
    # consider the skipping scheme & traversal through filters without
    # up / down sampling layers in nodes
    # [list(range(depth // 2)), depth // 2, list(reversed(range(depth // 2)))]
    # [list(range(depth//2)), depth//2, list(range(depth//2 + 1, depth))]

    # considering depth = 3, that nodes = [features, conv , pool, conv, deconv, conv]
    # skip-indexing needs to be adjusted:
    # [list(range(1,depth,2)), depth, list(reversed(range(1,depth,2)))]

    # rep = reps[f],inp = features, filt = filters[f], kernel = kernel

    # (ARCHITECHTURE) ---------------------------------------------------
    # tf.graph is stored in list, that extends with each graph operation,
    # facilitating the skipping schemes. therefore direct path through graph: input=nodes[-1]
    nodes = list()
    nodes.append(features)
    depth = len(filters)

    # (downward path)-----------------------------
    for f in range(depth // 2):

        nodes.append(convrep(
            mode=mode,
            rep=reps[f],
            inp=nodes[-1],
            filt=filters[f],
            ker=kernel))

        if sampling:
            nodes.append(tf.layers.max_pooling2d(
                inputs=nodes[-1],
                pool_size=2,
                strides=2,
                padding='VALID',
            ))
            # print(nodes[-1].shape, 'pool')

    nodes.append(convrep(
        mode=mode,
        rep=reps[depth // 2],
        inp=nodes[-1],
        filt=filters[depth // 2],
        ker=kernel))

    # (upward path) -------------------------------
    # note that while filters need to linearly traverse, where they left off,
    # the skipping index depends on the nodes scheme.
    for f, s in zip(range(depth // 2 + 1, depth), reversed(range(1, depth, 2))):
        # note skipindecies = range(depth // 2) if nodes was [conv, conv, conv]
        # but is [features, conv, pool, conv, deconv, conv] for depth = 3
        if sampling:
            nodes.append(convtransp(
                mode=mode,
                tensor_in=nodes[-1],
                filt=filters[f],
                ker=kernel))

        # print(nodes[s], nodes[-1], 'concat')
        if Uskip:
            print(nodes[s].shape, nodes[-1].shape, 'concat')
            skip = tf.concat([nodes[s], nodes[-1]], axis=3)
        else:
            skip = nodes[-1]

        nodes.append(convrep(
            mode=mode,
            rep=reps[f],
            inp=skip,
            filt=filters[f],
            ker=kernel))

    # (gather all information)-----------------------------
    prediction = tf.layers.conv2d(
        nodes[-1],
        filters=1,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None)

    if dncnn_skip: prediction = prediction + features

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

    # TRAINING
    # for TRAIN & EVAL
    loss = {
        'MAE': l1,
        'MSE': mse,
        'SSIM': ssim,
        'MS-SSIM': ssim_multiscale,
        'MS-SSIM-GL1': ssim_multiscale_gl1
    }[lossflavour](prediction, labels)

    loss = l1(prediction, labels)
    tf.summary.scalar("Value_Loss_Function", loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.histogram('Summary_final_layer', prediction)
        tf.summary.histogram('Summary_labels', labels)
        tf.summary.image('Input_Image', features, max_outputs=4)
        tf.summary.image('Output_Image', prediction, max_outputs=4)
        tf.summary.image('True_Image', labels, max_outputs=4)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            original_optimizer = tf.train.AdamOptimizer(
                learning_rate=0.03)
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
                # cannot write out SSIM, as eval_update_ops must be defined
            }
        )

    merged_summary = tf.summary.merge_all()
