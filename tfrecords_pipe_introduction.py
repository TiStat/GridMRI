'''All snippets are taken from https://www.tensorflow.org/guide/datasets'''

'''Every Pipeline comprises of two things:
a Dataset object (source) and an Iterator

(tf.data.Dataset) --------------------------------------------------------------
## 1) Abstract defintion
A Dataset is a sequence of Elements. An element contains one or more Tensors. These Tensors are called 'components' Each Tensor hast two attributes:
1) output_types
2) output_shape.
Each Tensor can have one or multiple nested Tensors with again the same structure.

Consider an image classification problem: An element contains two tensors:
A) the image Tensor type=float32, shape=(256\*256\*3)
B) a Label type=int, shape=1


## 2) Constructing Dataset
To CONSTRUCT a Dataset from some tensors in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices()
if your input data are on disk in the recommended TFRecord format, you can construct a tf.data.TFRecordDataset

dataset = tf.data.TFRecordDataset(filenames)
with filenames relating to one or more TFRecords. It is a tf.data.Dataset object

It's (nested) Tensors attributes are accessible:
print(dataset.output_types)
print(dataset.output_shapes)

## 3 Transform Dataset

elementwise transoformations
dataset.map()      e.g. .map(parser)


multi-element transoformations
dataset.batch()


(tf.data.Iterator)--------------------------------------------------------------
To consume values from a Dataset, an iterator object is constructed,
which provides access to one element of the dataset at a time:

iterator = dataset.make_one_shot_iterator()

A tf.data.Iterator provides two operations: Iterator.initializer,
which enables you to (re)initialize the iterator's state; and Iterator.get_next(),
which returns tf.Tensor

there are various types of iterators,
e.g. one_shot_iterator

'''

# (Iterate over Dataset.batches ONCE) ------------------------------------------
import tensorflow as tf

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# define dataset
dataset = tf.data.Dataset.range(100)
dataset = dataset.batch(10)

# define interator
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(10): # batch size
  value = sess.run(next_element)
  print(i, value)



# (Iterate through Dataset.batches for multiple EPOCHS) -----------------------
# REPEAT
# define dataset
dataset = tf.data.Dataset.range(100)
dataset = dataset.batch(10)
# repeating the dataset for multiple epochs (concatenate the dataset at end)
dataset = dataset.repeat(2) # repeat() would indicate infinite

# define interator
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 30 epochs.
for _ in range(3):
  sess.run(iterator.initializer)
  while True:
    try:
      print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      break
  print('end of epoch') # [Perform end-of-epoch calculations here.]

# (SAME but with shuffle) ------------------------------------------------------
# SHUFFLE
# define dataset
dataset = tf.data.Dataset.range(100)
dataset = dataset.repeat()
dataset = dataset.batch(10)
dataset = dataset.shuffle(2 ) # takes 2 complete batches and repositions them in their pipeorder!

# define interator
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(20): # batch size
  value = sess.run(next_element)
  print(i, value)



# (Dataset.from_tensor_slices(dict)) -------------------------------------------
dataset = tf.data.Dataset.from_tensor_slices(
   {# single dimension e.g. label: 4 observations
    "a": tf.random_uniform([4]),
    # for each label 'observation' there are 100 values
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)
   }
)
print(dataset.output_types)
print(dataset.output_shapes)

{"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)}


# (invoke operations on examples, while iterating) -----------------------------
## 1)  invoke operations on tensors while iterarate over exampes
# The Iterator.get_next() method returns one or more tf.Tensor objects that
# correspond to the symbolic next element of an iterator. Each time these
# tensors are evaluated, they take the value of the next element in the
# underlying dataset. Note: you must use the returned tf.Tensor objects in a
# TensorFlow expression, and pass the result of that expression to
# tf.Session.run() to get the next elements (i.e. evaluate the tensor)
# and advance the iterator:


dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(result))
  except tf.errors.OutOfRangeError:
    break


# (iterating in nested structures) ---------------------------------------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# create dataset with two tensor
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

# iterator definition
iterator = dataset3.make_initializable_iterator()
sess.run(iterator.initializer) # is required
# invoke iterator for the first time
next1, (next2, next3) = iterator.get_next()

# use example from iterator
print(next1, next2, next3)

# note that any invoke on either next1, next2, next3 will advance the
# iterator for all!
# consider the net run after each other
# print(sess.run(next1))
# print(sess.run(next1))
# print(sess.run(next1))
# print(sess.run(next1))
# print(sess.run(next1)) # out of range! as tf.random_uniform([4,10]) 4 is decicive here!

# print(sess.run(next2)) # also out of range due to previous advances!
# print(sess.run(next3))
