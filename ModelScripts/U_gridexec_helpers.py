import numpy as np
import tensorflow as tf

import pickle
import pandas as pd
from glob import glob
import random

def loadData(platform, root):
    # TODO replace with input pipeline on png
    if platform == 'hpc':
        train_data = np.load('/scratch2/truhkop/knee1/data/X_train.npy')
        train_labels = np.load('/scratch2/truhkop/knee1/data/Y_train.npy')

        test_data = np.load('/scratch2/truhkop/knee1/data/X_test.npy')
        test_labels = np.load('/scratch2/truhkop/knee1/data/Y_test.npy')

        # shorten predict / eval
        random.seed(42)
        randindex = random.sample(list(range(test_data.shape[0])), k=300)
        test_data = test_data[randindex]
        test_labels = test_labels[randindex]

    elif platform == 'cloud' or platform == 'home':
        # for development purposes only
        train_data = np.load(root + '/data/X_test_subset.npy')
        train_labels = np.load(root + '/data/Y_test_subset.npy')

        test_data = np.load(root + '/data/X_test_subset.npy')
        test_labels = np.load(root + '/data/Y_test_subset.npy')

        # shorten predict / eval
        random.seed(42)
        randindex = random.sample(list(range(test_data.shape[0])), k=16)
        test_data = test_data[randindex]
        test_labels = test_labels[randindex]

    print(train_data.shape)
    print(test_data.shape)

    return train_data, train_labels, test_data, test_labels


def benchit (candidatetensor, benchtensor):
    '''
     Benchmark comparison from x to bench y.
    :param candidatetensor: either test_data or benchtensor
    :param benchtensor: test_labels
    :return: tuple of arrays:
    Mean-absolute-error, Mean-squared-error, Structural Similarity Index
    '''
    candidatetensor = np.array([np.reshape(arr, (256, 256, 1)) for arr in candidatetensor])
    benchtensor = np.array([np.reshape(arr, (256, 256, 1)) for arr in benchtensor])

    MAE = np.array([np.mean(yi - xi) for xi, yi in zip(candidatetensor, benchtensor)])
    MSE = np.array([np.mean((yi - xi) ** 2) for xi, yi in zip(candidatetensor, benchtensor)])

    X = tf.Variable(candidatetensor)
    Y = tf.Variable(benchtensor)
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

    return MAE, MSE, SSIM


def pandasResults (root, parentdir, melt):
    ''':param parentdir: directory path to all gridexecutions
    :param melt: optionally also return the melted pandas dataframe for ease of plotting'''

    # (Resultfiles) ----------------------
    filenames = glob(root + parentdir + '/Result_*')
    GRID = list()
    for gridres in filenames:
        with open(gridres, "rb") as fp:  # Unpickling
            grid = pickle.load(fp)

            GRID.extend(grid)
    Gb = pd.DataFrame(GRID)
    Gb.filters = Gb.filters.astype(str)
    Gb.reps = Gb.reps.astype(str)

    Gb = Gb.dropna(subset=['name'])
    Gb = Gb.drop_duplicates(keep='first')

    # Descriptive info
    print('------------------ Results ----------------------')
    print('total train time: {} '.format(Gb.traintime.sum()))
    print(Gb.groupby('lossflavour')['lossflavour'].count())

    Gb.to_csv(root + parentdir + '/RESULTS.csv')

    if melt:
        meltedg = pd.melt(Gb, id_vars=['name', 'lossflavour', 'kernel', 'reps', 'filters', 'batch', 'trainsteps'],
                          value_vars=['M-MAE', 'M-MSE', 'M-SSIM'])

        meltedg.to_csv(root + parentdir + '/RESULTESmelt.csv')

    # (BENCHMARK files)  ------------------------
    filenames = glob(root + parentdir + '/benchmark_*')
    BENCH = pd.DataFrame()
    for file in filenames:
        with open(file, "rb") as fp:  # Unpickling
            b = pickle.load(fp)

            bench = pd.DataFrame(b[0])
            for i in range(1, len(b)):
                bench = bench.append(pd.DataFrame(b[i]))  # single img bench
            BENCH = BENCH.append(bench)

        # aggregated stat
        print('------------------ Bench ----------------------')
        print('MAE', BENCH.groupby('name')['MAE'].mean())
        print('MSE', BENCH.groupby('name')['MSE'].mean())
        print('SSIM', BENCH.groupby('name')['SSIM'].mean())

    BENCH.to_csv(root + parentdir + '/BENCH.csv')

    if melt:
        meltedBENCH = pd.melt(BENCH, id_vars=['name'],
                              value_vars=['MAE', 'MSE', 'SSIM'])
        meltedBENCH.to_csv(root + parentdir + '/BENCHmelt.csv')


