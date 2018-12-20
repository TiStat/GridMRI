import tensorflow as tf

def parser (record):
    '''function expects that the record file contains tuples containing x & y'''
    
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    features = {'x': tf.FixedLenFeature([], tf.string), # FIXME image was saved as string, but somehow float is expected
                'y': tf.FixedLenFeature([], tf.string)}

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=record,
                                             features=features)

    # Decode the raw bytes so it becomes a tensor with type.
    image_x = tf.decode_raw(parsed_example['x'], tf.float32)
    image_y = tf.decode_raw(parsed_example['y'], tf.float32)

    image_x = tf.reshape(image_x, [256,256,1]) # FIXME dimension cause error
    image_y = tf.reshape(image_y, [256, 256, 1])

    # is reshape of image required?

    return image_x, image_y  # ggf in einem dict?


def input_fn (filenames):
    '''carefull map_and_batch & shuffle_and_repeat will soon be depreciated!
    instead: https://www.tensorflow.org/api_docs/python/tf/contrib/data/shuffle_and_repeat
         https://www.tensorflow.org/api_docs/python/tf/contrib/data/map_and_batch
    '''
    # generate a source DATASET OBJECT
    dataset = tf.data.TFRecordDataset(filenames=filenames) #, num_parallel_reads=40)  # unsure about this parameter
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1024, 1))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parser, 32))  # this soon will be DEPRECIATED

    # apply per element transformations (map)
    # dataset = dataset.map(parser, num_parallel_calls=12)

    # consecutive elements are returned (batch [a multi element transformation])
    # dataset = dataset.batch(batch_size=1000)

    dataset = dataset.prefetch(buffer_size=2)
    return dataset

# Pipelines
def train_input_fn():
    return input_fn(filenames=["C:/Users/timru/Documents/CODE/deepMRI1/TFRecords/XGray.tfrecord"]) #,
                             # "/home/jovyan/Data/TFRecords/short_Grayvalid.tfrecords"])
# the second filename is just for convenience to look, if multiple tfrecords are read in correctly
# from a model's perspective, this is absolute nonsense!

def val_input_fn():
    return input_fn(filenames=["./short_Grayvalid.tfrecords"])



# Setting up the model and converting it to an estimator API:
# ImpNet = tf.estimator.Estimator(config=runconf,
#                                 model_fn=cnn_model_fn,
#                                 model_dir="/tmp/tmp/Gaussian")

# And training the model using the pipe:
# ImpNet.train(
#     input_fn=train_input_fn,
#     steps=1)









# (HINT: Consuming TFRecords with TRAIN & VALIDATION) --------------------------
'''### Stream over Tfrecords
The tf.data API supports a variety of file formats so that you can
process large datasets that do not fit in memory. For example,
the TFRecord file format is a simple record-oriented binary format
that many TensorFlow applications use for training data.
The tf.data.TFRecordDataset class enables you to stream over the
contents of one or more TFRecord files as part of an input pipeline

#Dataset Definition
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)

#iterate over dataset


### Splitting TFRecords in training & validation
The filenames argument to the TFRecordDataset initializer can either
be a string, a list of strings, or a tf.Tensor of strings.
Therefore if you have two sets of files for TRAINING and VALIDATION
purposes, you can use a tf.placeholder(tf.string) to represent
the filenames, and initialize an iterator from the appropriate filenames:


#(CODE SNIPPED) ----------------------------------------------------------

#variable dataset definition
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

#You can feed the initializer with the appropriate filenames for the current
#phase of execution, e.g. training vs. validation.

#Initialize `iterator` with training data.
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

#Initialize `iterator` with validation data.
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
 ----------------------------------------------------------------------------
## Using .map()
### Parsing tf.Example protocol buffer
Many input pipelines extract tf.train.Example protocol buffer messages from a TFRecord-format file (written, for example, using tf.python_io.TFRecordWriter). Each tf.train.Example record contains one or more "features", and the input pipeline typically converts these features into tensors.

#Transforms a scalar string `example_proto` into a pair of a scalar string and
#a scalar integer, representing an image and its label, respectively.
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

#Creates a dataset that reads all of the examples from two files, and extracts
#the image and label features.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)

-------------------------------------------------------------------------------
# Training workflow with tf.estimator.Estimator API
To use a Dataset in the input_fn of a tf.estimator.Estimator, simply return the Dataset and the framework will take care of creating an iterator and initializing it for you.

def dataset_input_fn():
  filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
  dataset = tf.data.TFRecordDataset(filenames)

  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    #image = tf.image.decode_jpeg(parsed["image_data"])
    #image = tf.reshape(image, [299, 299, 1])
    #label = tf.cast(parsed["label"], tf.int32)

    # note that image and date_time are a pair-feature;
    # i.e. they are one tensor with two nested tensors and therefore
    # passed jointly in a dict. However, label is the resulting y obje and passed
    # seperately.
    return {"image_data": image, "date_time": parsed["date_time"]}, label

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)

  # Each element of `dataset` is tuple containing a dictionary of features
  # (in which each value is a batch of values for that feature), and a batch of
  # labels.
  return dataset
'''



'''
import numpy as np
import tensorflow as tf

# Generate TFRecords
def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def createRecord (input_path, output_path, Pindexrange=range(1, 21)):
    
    #input_path: path of P[i]_X.npy & P[i]_Y.npy arrays
    #output_path: is restricted to current wd; only name of output can be specified
    #Pindexrange: range of Patient-array 'P[i]' considered. this becomes relevant, as the data is split into test & validation
    #n is size of sample on images. Set to length of array (4055) of images for full conversion
 
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for i in Pindexrange:
            # note: this for loop must be changed to accomodate for split of dataset
            print("Starting to write P[" + str(i) + "] image array out!")

            # also actually a task performed by DATASET.zip, after having X & Y
            # in sepperate TFRecord files? therefore, they could be zipped?
            # Create the path of the current input nd-array
            path_X = input_path + 'P' + str(i) + "_X.npy"
            path_Y = input_path + 'P' + str(i) + "_Y.npy"

            # Load X,Y
            X = np.load(path_X)
            Y = np.load(path_Y)

            # Transform them to tf.records. iterate over image number dimension.
            for x, y  in zip(X, Y):
                example = tf.train.Example(features=tf.train.Features(
                    feature=
                    {
                        'x': _float_feature(x.tostring()),
                        'y': _float_feature(y.tostring())
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
createRecord(input_path = "C:/Users/timru/Documents/CODE/deepMRI1/Data/",
             output_path =  "C:/Users/timru/Documents/CODE/deepMRI1/TFRecords/XGray.tfrecord",
             Pindexrange = range(1,3))



def parser (record):
    '''function expects that the record file contains tuples containing x & y'''
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    features = {'x': tf.FixedLenFeature([], tf.string),
                'y': tf.FixedLenFeature([], tf.string)}

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=record,
                                             features=features)

    # Decode the raw bytes so it becomes a tensor with type.
    image_x = tf.decode_raw(parsed_example['x'], tf.float32)
    image_y = tf.decode_raw(parsed_example['y'], tf.float32)


    return image_x, image_y  # ggf in einem dict?


def input_fn (filenames):
    '''carefull map_and_batch & shuffle_and_repeat will soon be depreciated!
    instead: https://www.tensorflow.org/api_docs/python/tf/contrib/data/shuffle_and_repeat
         https://www.tensorflow.org/api_docs/python/tf/contrib/data/map_and_batch
    '''
    # generate a source DATASET OBJECT
    dataset = tf.data.TFRecordDataset(filenames=filenames,
                                      num_parallel_reads=40)  # unsure about this parameter
    # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1024, 1))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parser, 32))  # this soon will be DEPRECIATED

    # apply per element transformations (map)
    # dataset = dataset.map(parser, num_parallel_calls=12)

    # consecutive elements are returned (batch [a multi element transformation])
    # dataset = dataset.batch(batch_size=1000)

    #dataset = dataset.prefetch(buffer_size=40)
    return dataset


D = input_fn('C:/Users/timru/Documents/CODE/deepMRI1/TFRecords/XGray.tfrecord')

tf.enable_eager_execution()
tf.executing_eagerly()
raw_dataset = tf.data.TFRecordDataset('C:/Users/timru/Documents/CODE/deepMRI1/TFRecords/XGray.tfrecord')
for raw_record in raw_dataset.take(1):
  print(repr(raw_record))

'''