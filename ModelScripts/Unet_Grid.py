import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.image_ops_impl import _fspecial_gauss

#tf.enable_eager_execution()
#tf.executing_eagerly()

platform = ['hpc', 'cloud', 'home'][2]
root = {
    'hpc':'/scratch2/truhkop/',
    'cloud':'/home/cloud/',
    'home':'C:/Users/timru/Documents/CODE/deepMRI1/'} [platform]

# (DATA) -----------------------------------------------------------------------
if platform == 'hpc':
    train_data = np.load('/scratch2/truhkop/knee1/data/X_train.npy')
    train_labels = np.load('/scratch2/truhkop/knee1/data/Y_train.npy')

    test_data= np.load('/scratch2/truhkop/knee1/data/X_test.npy')
    test_labels=  np.load('/scratch2/truhkop/knee1/data/Y_test.npy')

elif platform == 'cloud':
    train_data = np.load(root +'/data/X_test_subset.npy')
    train_labels = np.load(root + '/data/Y_test_subset.npy')

    test_data = np.load(root +'/data/X_test_subset.npy')
    test_labels = np.load(root + '/data/Y_test_subset.npy')
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

    train_data, train_labels = subset_arr(X, Y, batchind=range(20))
    test_data, test_labels = subset_arr(X, Y, batchind=range(5, 8))


# (Benchmark) ------------------------------------------------------------------
print('starting Benchmarking')
MAE = np.array([np.mean(yi - xi) for xi, yi in zip(test_data, test_labels)])
MSE = np.array([np.mean((yi - xi)**2) for xi, yi in zip(test_data, test_labels)])

X = tf.Variable(test_data)
Y = tf.Variable(test_labels)
Z = tf.stack([X, Y], axis=1)
SSIM = tf.map_fn(lambda x: tf.image.ssim(img1 = x[0], img2 = x[1], max_val =1), Z, dtype=tf.float32)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    SSIM = sess.run(SSIM)

with open("benchmark.txt", "wb") as fp:  # pickling
    pickle.dump(pd.DataFrame({'name':'benchmark','MAE': MAE, 'MSE': MSE, 'SSIM': SSIM}), fp)

# (Grid formulation) -----------------------------------------------------------
with open("grid.txt", "rb") as fp:  # Unpickling
    grid = pickle.load(fp)

pd.DataFrame(grid)
# pd.DataFrame(grid)
# setting up the parameter environment, upon which model_fn harvests
for config in grid:
    kernel = config['kernel']
    filters = config['filters']
    lossflavour = config['lossflavour']
    batch = config['batch']
    reps = config['reps']
    dncnn_skip = config['dncnn_skip']
    trainsteps = config['trainsteps']

    if len(filters) != reps:
        print('fucked up dimensions')
    if len(filters) % 2 != 1:
        print('fucked up uneven length')

    def Unet_model_fn (features, labels, mode):
        ''' :kernel: kernelsize, skalar parameter for conv kernel,
            :filters: list, specifying for all convs in a conv_rep block, how many output filters (kernel) are to be
            applied
            :lossflavour: string, specifying which loss function is to be used. MS-SSIM, MS-SSIM-GL1 are not tested
            eagerly. Can be any of: ['MSE', 'MAE', 'SSIM', 'MS-SSIM','MS-SSIM-GL1']
            :batch: size of the batch, applied to all conv layers
            :reps: list, giving the number of stacked convs in the respective conv_rep block
            :dncnn_skip: boolean, optional skip connectionen from input (x) to prediction. (p = x + netoutput)
            :trainsteps: integer, number of training steps for all model configurations
        '''

        # nodes of tf.graph, stored in list, that extends with each operation
        # therefore direct path: input=nodes[-1]
        nodes = list()
        nodes.append(features)
        depth = len(filters)

        # consider the skipping scheme & traversal through filters without
        # up / down sampling layers in nodes
        # [list(range(depth // 2)), depth // 2, list(reversed(range(depth // 2)))]
        # [list(range(depth//2)), depth//2, list(range(depth//2 + 1, depth))]

        # considering depth = 3, that nodes = [features, conv , pool, conv, deconv, conv]
        # skip-indexing needs to be adjusted:
        #[list(range(1,depth,2)), depth, list(reversed(range(1,depth,2)))]

        # downward path
        for f in range(depth // 2):
            nodes.append(tf.contrib.layers.repeat(
                inputs=nodes[-1], # always takes previous conv2d
                num_outputs=filters[f],
                repetitions=reps[f],
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
            num_outputs=filters[depth // 2],
            repetitions=reps[depth // 2],
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
        # note that while filters need to linearly traverse, where it left off,
        # the skipping index depends on the nodes scheme.
        for f, s in zip(range(depth//2 + 1, depth), reversed(range(1,depth,2))):
            # note skipindecies = range(depth // 2) if nodes was [conv, conv, conv]
            # but is [features, conv, pool, conv, deconv, conv] for depth = 3
            nodes.append(tf.contrib.layers.conv2d_transpose(
                inputs=nodes[-1],
                kernel_size=2,
                stride=1,
                num_outputs=filters[f],
                padding='VALID',
                activation_fn=tf.nn.relu,
                normalizer_fn=tf.layers.batch_normalization,
                normalizer_params={
                    'momentum': 0.99,
                    'epsilon': 0.001,
                    'trainable': False,
                    'training': mode == tf.estimator.ModeKeys.TRAIN},
            ))

            skip = tf.concat([nodes[s], nodes[-1]], axis=3)

            nodes.append(tf.contrib.layers.repeat(
                inputs=skip,
                num_outputs=filters[f],
                repetitions=reps[f],
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
            inputs=nodes[-1],
            num_outputs=1,
            kernel_size=kernel,
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu
        )

        if dncnn_skip:
            prediction = prediction + features

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

        def ssim_multiscale (prediction, label):
            return 1 - tf.reduce_mean(
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
            tf.summary.image('Input_Image', features, max_outputs = 4)
            tf.summary.image('Output_Image', prediction, max_outputs = 4)
            tf.summary.image('True_Image', labels, max_outputs = 4)
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


    # (ESTIMATORS) -------------------------------------------------------------
    d = datetime.now()
    modelID = 'Unet_{}_{}_{}_{}_{}'.format(lossflavour, d.month, d.day, d.hour, d.minute, d.second)
    Unet = tf.estimator.Estimator(
        model_fn=Unet_model_fn,
        model_dir=root + 'grid/' + modelID,
        config=tf.estimator.RunConfig(
            save_summary_steps=20,
            log_step_count_steps=20))

    print('generated {} Estimator  ----------------------------------------------'.format(modelID))
    print('Config: {}'.format(config))

    # (TRAINING) ---------------------------------------------------------------
    start_time = datetime.now()
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=batch,
        num_epochs=None,
        shuffle=True)

    Unet.train(input_fn=train_input_fn, steps=trainsteps)

    # write out to grid.txt note the view!
    time_elapsed = datetime.now() - start_time
    print('Trained {}'.format(modelID))
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))



    # (EVALUATE) ---------------------------------------------------------------
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_data,
        y=test_labels,
        batch_size=1,
        num_epochs=1,
        shuffle=False)

    # evaluation for each prediction stored in benchmark.txt
    predictions = list(Unet.predict(input_fn=test_input_fn))
    print('predicted {}'.format(modelID))

    MAE = np.array([np.mean(yi - xi) for xi, yi in zip(test_data, predictions)])
    MSE = np.array([np.mean((yi - xi) ** 2) for xi, yi in zip(test_data, predictions)])

    predictions = [np.reshape(arr, (1, 256, 256, 1)) for arr in predictions]
    predictions = np.vstack(predictions)

    X = tf.Variable(test_data)
    Y = tf.Variable(predictions)
    Z = tf.stack([X, Y], axis=1)
    SSIM = tf.map_fn(
        lambda x: tf.image.ssim(
            img1=x[0],
            img2=x[1],
            max_val=1),
        Z, dtype=tf.float32)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        SSIM = sess.run(SSIM)

    with open("benchmark.txt", "rb") as fp:  # Unpickling
        bench = pickle.load(fp)
    newmodel = pd.DataFrame({'name': modelID, 'MAE': MAE, 'MSE': MSE, 'SSIM': SSIM})
    with open("benchmark.txt", "wb") as fp:  # pickling
        pickle.dump(pd.concat([bench, newmodel]), fp)
        # note that benchmark.txt contains all MAE, MSE, SSIM for all predictions

    # aggregate the results and write to grid.txt note the view on config
    config['name'] = modelID
    config['traintime'] = time_elapsed
    config['M-MAE'] = np.mean(MAE)
    config['M-MSE'] = np.mean(MSE)
    config['M-SSIM'] = np.mean(SSIM)
    print('Evaluated {}'.format(modelID))