

# (PNG pipe) -------------------------------------------------------------------
import scipy.misc as sci
import numpy as np
import tensorflow as tf
import glob
import os


# writing npy to image files
img = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P1_X.npy')
for i, im in enumerate(img):
    sci.imsave('C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/P1_{}X.jpg'.format(i), im)

img = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P1_Y.npy')
for i, im in enumerate(img):
    sci.imsave('C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/P1_{}Y.jpg'.format(i), im)


# ------------------------------------------------------------------------------
# (try to split on filename basis)
Xnames = []
Ynames = []
# TODO replicate the Y file names 5 times
Xnames.extend(glob.glob(os.path.join('C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/', '*X.jpg')))
Ynames.extend(glob.glob(os.path.join('C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/', '*Y.jpg')))

zippednames = [(x,y) for x,y in zip(Xnames, Ynames) ]

# at this point one can split the dataset into train, test , valid, eval
train_set = zippednames[0:3000]
test_set = zippednames[3000:4055]

def input_fn(zippedfilepaths):
    ''':zippedfilepaths: list of tuples, which contain the filepaths to (featureimg, labelimg_path)'''

    zippednames = tf.data.Dataset.from_tensor_slices(zippedfilepaths)
    zippednames = zippednames.shuffle(buffer_size=4055)

    image_dataset = zippednames.map(
        lambda x:
        (tf.image.decode_image(tf.read_file(x[0]), channels=1),
         tf.image.decode_image(tf.read_file(x[1]), channels=1))  # TODO zippednames outshape (256,256,1) reshape?
    )  # decode_png

    return image_dataset


image_dataset = input_fn(train_set)
iterator = image_dataset.make_one_shot_iterator()
next_image = iterator.get_next()
with tf.Session() as sess:
    X, Y = sess.run(next_image)


# ------------------------------------------------------------------------------

filename_datasetX = tf.data.Dataset.list_files("C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/*X.jpg")
filename_datasetY = tf.data.Dataset.list_files("C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/*Y.jpg")


# Make a Dataset of image tensors by reading and decoding the files.
image_datasetX = filename_datasetX.map(lambda x: tf.image.decode_image(tf.read_file(x), 1)) # decode_png
image_datasetY = filename_datasetY.map(lambda x: tf.image.decode_image(tf.read_file(x), 1))

full = tf.data.Dataset.zip((image_datasetX, image_datasetY))

iterator = full.make_one_shot_iterator()
next_image = iterator.get_next()

with tf.Session() as sess:
    X, Y = sess.run(next_image)

# ------------------------------------------------------------------------------
Xnames = []
Ynames = []
Xnames.extend(glob.glob(os.path.join('C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/', '*X.jpg')))
Ynames.extend(glob.glob(os.path.join('C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/', '*Y.jpg')))

Xnames = tf.data.Dataset.from_tensors(tf.convert_to_tensor(Xnames))
Ynames = tf.data.Dataset.from_tensors(tf.convert_to_tensor(Ynames))

image_datasetX = Xnames.map(lambda x: tf.image.decode_image(tf.read_file(x), 1))  # decode_png
image_datasetY = Ynames.map(lambda x: tf.image.decode_image(tf.read_file(x), 1))

full = tf.data.Dataset.zip((image_datasetX, image_datasetY))

iterator = full.make_one_shot_iterator()
next_image = iterator.get_next()
with tf.Session() as sess:
    X, Y = sess.run(next_image)

    #types = ('*X.jpg', '*Y.jpg')
    #images_list = []
    #for files in types:
    #    images_list.extend(glob.glob(os.path.join('C:/Users/timru/Documents/CODE/deepMRI1/Datajpg/', files)))







###############################################################################
###############################################################################
###############################################################################
#(Failing attempt) memory usage is exceeded

import tensorflow as tf
import numpy as np

# initialize
X = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P1_X.npy')
Y = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P1_Y.npy')
dataset0 = tf.data.Dataset.from_tensor_slices((X, Y))

for i in [2,3]:
    X = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P{}_X.npy'.format(i))
    Y = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P{}_Y.npy'.format(i))
    dataset1 = tf.data.Dataset.from_tensor_slices((X,Y))
    dataset0 = dataset0.concatenate(dataset1)

iterator = dataset0.make_one_shot_iterator()
next_image = iterator.get_next()
with tf.Session() as sess:
    imgX, imgY = (sess.run(next_image))

X = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P1_X.npy')
imgX == X[0]




###############################################################################
###############################################################################
###############################################################################





'''Tutorial taken and edited from
https://www.tensorflow.org/guide/datasets_for_estimators
'''

import tensorflow as tf
import numpy as np

def read_npy_file(item):
    data = np.load(item.decode())
    return data.astype(np.float32)

filename_datasetX = tf.data.Dataset.list_files("C:/Users/timru/Documents/CODE/deepMRI1/Data/*X.npy")
filename_datasetY = tf.data.Dataset.list_files("C:/Users/timru/Documents/CODE/deepMRI1/Data/*Y.npy")

imgX = filename_datasetX.map(lambda item: tf.py_func(read_npy_file, [item], [tf.float32,]))
imgY = filename_datasetY.map(lambda item: tf.py_func(read_npy_file, [item], [tf.float32,]))
full = tf.data.Dataset.zip((imgX,imgY)) # results in a tuple
#full = full.shuffle(2)

iterator = full.make_one_shot_iterator()
next_image = iterator.get_next()

# Start a new session to show example output.
with tf.Session() as sess:
    X, Y = sess.run(next_image) # carefull: shape [1,4055,256,256]
    print(X)

    print(tf.shape(X))
    X, Y = sess.run(next_image)
    print(X)

import matplotlib.pyplot as plt

plt.imshow(np.reshape(X[0][0], (256, 256)) * 255, cmap='gray')
plt.show()







# ------------------------------------------------------------------------------

# (parallel iterating on two Datasets)------------------------------------------
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
    print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
    print(sess.run(next_element))

# (placeholder approach TF_CODE SNIPPED, not working, WITH! manager) -----------
# original data is one array, with names
# Load the training data into two NumPy arrays, for example using `np.load()`.
# Does your data fit into memory? If so, you can follow the instructions from
# the Consuming NumPy Arrays section of the docs
with np.load("/var/data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
# dataset = ...
iterator = dataset.make_initializable_iterator()

with tf.Session as sess:
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                              labels_placeholder: labels})

'C://Users//timru//Documents//CODE//deepMRI1//Data//P1_X.npy'.replace('//', '/')

# (feed_dict for placeholders example) -----------------------------------------
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# Loop forever, alternating between training and validation.
# Run 200 steps using the training dataset. Note that the training dataset is
# infinite, and we resume from where we left off in the previous `while` loop
# iteration.
for _ in range(200):
    print(sess.run(next_element, feed_dict={handle: training_handle}))

# Run one pass over the validation dataset.
sess.run(validation_iterator.initializer)
for _ in range(50):
    print(sess.run(next_element, feed_dict={handle: validation_handle}))

# (feed_dict -> fetches -> placeholder) ----------------------------------------
W = tf.constant([10, 100], name='const_W')

# these placeholders can hold tensors of any shape
# we will feed these placeholders later
x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

# tf.multiply is simple multiplication and not matrix
Wx = tf.multiply(W, x, name="Wx")
y = tf.add(Wx, b, name='y')

with tf.Session() as sess:
    '''all the code which require a session is writer here
    here Wx is the fetches parameter. fetches refers to the node of the graph we want to compute
    feed_dict is used to pass the values for the placeholders
    '''
    print("Intermediate result Wx: ", sess.run(Wx, feed_dict={x: [3, 33]}))
    print("Final results y: ", sess.run(y, feed_dict={x: [5, 50], b: [7, 9]}))

writer = tf.summary.FileWriter('./fetchesAndFeed', sess.graph)
writer.close()

# (feed_dict read in dataset) --------------------------------------------------
# https://www.learningtensorflow.com/lesson4/
x = tf.placeholder("float", [None, 3])
y = x * 2

with tf.Session() as session:
    x_data = [[1, 2, 3],
              [4, 5, 6], ]  # maybe iteratively go over all filenames
    result = session.run(y, feed_dict={x: x_data})
    print(result)

# (feed_dict read in datasets one after the other) -----------------------------
x = tf.placeholder(tf.float32, None)  # tensor of any size can be read in
y = x * 2  # /255

filenames = ['./ArraysGray/P1_X.npy', './ArraysGray/P1_Y.npy']
with tf.Session() as session:
    for file in filenames:
        x_data = np.load(file)  # maybe iteratively go over all filenames
        result = session.run(y, feed_dict={x: x_data})
        print('printing ' + file)
        print(result)

# (feed_dict parallel traversal) -----------------------------------------------
root = 'C:/Users/timru/Documents/CODE/deepMRI1/Data/'
placeX = tf.placeholder(tf.float32, None)  # tensor of any size can be read in
placeY = tf.placeholder(tf.float32, None)
y = placeX / 255  # /255 # or any function on placeX
z = placeY * 3  # *3 #  # or any function on placeY

Xfilenames = ['P1_X.npy']
Yfilenames = ['P1_Y.npy']
with tf.Session() as session:
    for Xfile, Yfile in zip(Xfilenames, Yfilenames):
        X_data = np.load(root + Xfile)  # maybe iteratively go over all filenames
        Y_data = np.load(root + Yfile)
        result1 = session.run(y, feed_dict={placeX: X_data[10]})
        result2 = session.run(z, feed_dict={placeY: Y_data[10]})
        print('printing ' + Xfile)
        print(result1)

        print('printing ' + Yfile)
        print(result2)

# ------------------------------------------------------------------------------
# with all images.npy individually stored, and seperatelay stored labels & data
# https://stackoverflow.com/questions/50216747/how-to-implement-multi-threaded-import-of-numpy-arrays-stored-on-disk-as-dataset?answertab=active#tab-top
import tensorflow as tf
import numpy as np
import os


def gen ():
    inputs_path = ""
    labels_path = ""
    for input_file, label_file in zip(os.listdir(inputs_path), os.listdir(labels_path)):
        x = np.load(os.path.join(inputs_path, input_file))
        y = np.load(os.path.join(labels_path, label_file))
        yield x, y


INPUT_SHAPE = []
LABEL_SHAPE = []

# Input pipeline
ds = tf.data.Dataset.from_generator(
    gen, (tf.float32, tf.int64), (tf.TensorShape(INPUT_SHAPE), tf.TensorShape(LABEL_SHAPE)))
ds = ds.batch(8)
ds_iter = ds.make_initializable_iterator()
inputs_batch, labels_batch = ds_iter.get_next()


# (zip datasets) ---------------------------------------------------------------
# in order to keep track of the two datasets X & Y:
# # NOTE: The following examples use `{ ... }` to represent the
# # contents of a dataset.
# a = { 1, 2, 3 }
# b = { 4, 5, 6 }
# c = { (7, 8), (9, 10), (11, 12) }
# d = { 13, 14 }
#
# # The nested structure of the `datasets` argument determines the
# # structure of elements in the resulting dataset.
# Dataset.zip((a, b)) == { (1, 4), (2, 5), (3, 6) }
# Dataset.zip((b, a)) == { (4, 1), (5, 2), (6, 3) }
#
# # The `datasets` argument may contain an arbitrary number of
# # datasets.
# Dataset.zip((a, b, c)) == { (1, 4, (7, 8)),
#                             (2, 5, (9, 10)),
#                             (3, 6, (11, 12)) }
#
# # The number of elements in the resulting dataset is the same as
# # the size of the smallest dataset in `datasets`.
# Dataset.zip((a, d)) == { (1, 13), (2, 14) }


np.load()
# (working version of that) ----------------------------------------------------
# https://github.com/keras-team/keras/issues/9707

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# how to get generator shuffle with generator.
def gen ():
    input_root = 'C:/Users/timru/Documents/CODE/deepMRI1/Data/'
    # hier muss über die P..._X und P.._Y gezipt und geloopt werden!
    for xin, yin in zip(['P1_X.npy', 'P2_X.npy'], ['P1_y.npy', 'P1_y.npy']):
        x = np.load(input_root + xin)
        y = np.load(input_root + yin)
        yield x[0:16], y[0:16]


INPUT_SHAPE = [16, 256, 256]
LABEL_SHAPE = [16, 256, 256]

# Input pipeline
ds = tf.data.Dataset.from_generator(
    gen, (tf.float32, tf.int64), (tf.TensorShape(INPUT_SHAPE), tf.TensorShape(LABEL_SHAPE)))
ds = ds.batch(1)  # is the first shape: [?, 16,256,256]
ds_iter = ds.make_one_shot_iterator()
inputs_batch, labels_batch = ds_iter.get_next()


# (generator & class & model class) --------------------------------------------
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0


# (iterators & NumPype9 --------------------------------------------------------
# https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
