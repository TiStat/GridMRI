import tensorflow as tf

def parser (record):
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

    # is reshape of image required?

    return {'x': image_x, 'y': image_y}  # ggf in einem dict?


def input_fn (filenames):
    '''carefull map_and_batch & shuffle_and_repeat will soon be depreciated!
    instead: https://www.tensorflow.org/api_docs/python/tf/contrib/data/shuffle_and_repeat
         https://www.tensorflow.org/api_docs/python/tf/contrib/data/map_and_batch
    '''
    # generate a source DATASET OBJECT
    dataset = tf.data.TFRecordDataset(filenames=filenames,
                                      num_parallel_reads=40)  # unsure about this parameter
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
    return input_fn(filenames=["/home/jovyan/Data/TFRecords/short_Graytrain.tfrecords"]) #,
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
