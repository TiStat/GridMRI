'''interesting article to convert various image formats to tfrecords:
https://planspace.org/20170403-images_and_tfrecords/

article on blowing-up TFRecords size when converting:
https://medium.com/coinmonks/storage-efficient-tfrecord-for-images-6dc322b81db4
'''

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

            # Create the path of the current input nd-array
            path_X = input_path + 'P' + str(i) + "_X.npy"
            path_y = input_path + 'P' + str(i) + "_Y.npy"

            # Load X,Y
            X = np.load(path_X)
            Y = np.load(path_y)

            # optional snipped:
            # Select only n images per input!
            select = np.random.choice(np.arange(0, 4055), size=n)
            X = X[select, :, :]
            Y = Y[select, :, :]

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