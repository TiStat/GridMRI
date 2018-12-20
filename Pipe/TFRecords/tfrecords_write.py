'''interesting article to convert various image formats to tfrecords:
https://planspace.org/20170403-images_and_tfrecords/

article on blowing-up TFRecords size when converting:
https://medium.com/coinmonks/storage-efficient-tfrecord-for-images-6dc322b81db4
'''

'''The idea of tensorflow pipeline with TFRecords in combination with Dataset api 
is to first put all the numpy arrays in the TFRecord file format, since .npy will 
have to be loaded entirely (every single time to extract some slices of it), which is costly.
i suppose, that tfrecords can be accessed partially? (found the answer YES:
https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)
and need not be read entirely.
Next up, putting the TFRecord created into a Dataset allows for appending this 
dataset with multiple Datasets or even zip them.
Therefor, all npy should be converted to a tfrecord as a whole 
(i.e. X & Y loaded and converted separately ), not only a subset of them!
later on, they should be zipped together as a dataset in tupel format (x,y).
The advantaage is that the entire dataset containing all zipped datasets is not read 
in memory all at once, but read in (and distributed) quickly from the iterator 
(to the required batch size), which is placed on the dataset,
which itself is merely a reference to all stored files.'''
'''it seems to be not as easy with zipping as i thought.
another way could by tupling (x, y) in a TFRecord and then batch_join over multiple TFRecords.
also look at batching & batch_join from https://www.tensorflow.org/api_guides/python/reading_data
in fact, this is depreciated, and will be removed by queuerunner: 
tf.data.Dataset.interleave(...).shuffle(min_after_dequeue).batch(batch_size)'''
import numpy as np
import tensorflow as tf

# Generate TFRecords
def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def createRecord (input_path, output_path, Pindexrange=range(1, 21), n=4055):
    '''
    input_path: path of P[i]_X.npy & P[i]_Y.npy arrays
    output_path: is restricted to current wd; only name of output can be specified
    Pindexrange: range of Patient-array 'P[i]' considered. this becomes relevant, as the data is split into test & validation
    n is size of sample on images. Set to length of array (4055) of images for full conversion
    '''
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for i in Pindexrange:
            # note: this for loop must be changed to accomodate for split of dataset
            print("Starting to write P[" + str(i) + "] image array out!")

            # also actually a task performed by DATASET.zip, after having X & Y
            # in sepperate TFRecord files? therefore, they could be zipped?
            # Create the path of the current input nd-array
            path_X = input_path + 'P' + str(i) + "_X.npy"
            path_y = input_path + 'P' + str(i) + "_Y.npy"

            # Load X,Y
            X = np.load(path_X)
            Y = np.load(path_y)

            # optional snipped:
            # Select only n images per input!
            # actually shuffeling is a task that is performed on Dataset!
            # so this snipped should be erased!
            select = np.random.choice(np.arange(0, 4055), size=n)
            X = X[select, :, :]
            Y = Y[select, :, :]

            # this reading format actually expected the X.npy to contain both x & y variable
            # that is it expected X to be an array of nested tuples (x ,y) which it does not.
            # therefore it must fail contentwise.

            # Transform them to tf.records. iterate over image number dimension.
            for i in range(X.shape[0]):
                example = tf.train.Example(features=tf.train.Features(
                    feature=
                    {
                        'x': _float_feature(X[i].tostring()),
                        'y': _float_feature(Y[i].tostring())
                    }
                )
                )
                # given the 'tupel', serialize this example
                writer.write(example.SerializeToString())
                if i % 5 == 0:  # eddit size to observe progress!
                    print('writing {}th image'.format(i))
            print("number of saved images: " + str(i + 1))

# carefull: writing is only possible to the dir where this script resides.
# input is not as restricted
createRecord(input_path = "../Cubes/ArraysGray/",
             output_path =  "short_Graytrain.tfrecord",
             Pindexrange = range(1,3), # range(1,4)
             n = 10)
createRecord(input_path = "../Cubes/ArraysGray/",
             output_path =  "short_Grayvalid.tfrecord",
             Pindexrange = range(3,4),
             n = 10)