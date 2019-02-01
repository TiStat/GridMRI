import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
# tf.enable_eager_execution()

imgsX = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P1_X.npy')
imgsY = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P1_Y.npy')
img1 = tf.reshape(imgsX[0], (1,256,256,1))
img2 = tf.reshape(imgsY[0], (1,256,256,1))

# ------------------------------------------------------------------------------
# kernel used internally of MS-SSIM
# https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/image_ops_impl.py

def _fspecial_gauss(size, sigma):
  """Function to mimic the 'fspecial' gaussian MATLAB function."""
  size = ops.convert_to_tensor(size, dtypes.int32)
  sigma = ops.convert_to_tensor(sigma)

  coords = math_ops.cast(math_ops.range(size), sigma.dtype)
  coords -= math_ops.cast(size - 1, sigma.dtype) / 2.0

  g = math_ops.square(coords)
  g *= -0.5 / math_ops.square(sigma)

  g = array_ops.reshape(g, shape=[1, -1]) + array_ops.reshape(g, shape=[-1, 1])
  g = array_ops.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
  g = nn_ops.softmax(g)
  return array_ops.reshape(g, shape=[size, size, 1, 1])

filter_size = constant_op.constant(11, dtype=dtypes.int32)
filter_sigma = constant_op.constant(1.5, dtype=img1.dtype)
# values according to MS-SSIM source code
kernel = _fspecial_gauss(
    size=filter_size,
    sigma=filter_sigma)

def loss_ssim_multiscale_GL1(trueimg, noisyimg, alpha = 0.84):
    ''' Loss function, calculating alpha * Loss_msssim + (1-alpha) gaussiankernel * L1_loss
    according to 'Loss Functions for Image Restoration with Neural Networks' [Zhao]
    :alpha: default value accoording to paper'''
    kernelonL1 = tf.nn.conv2d(
        input=tf.subtract(trueimg, noisyimg),
        filter=kernel,
        strides=[1, 1, 1, 1],  # according to MS-SSIM source
        padding='VALID')

    # number of patches * number of pixels per patch
    img_patch_norm = tf.to_float(kernelonL1.shape[1] * filter_size ** 2)
    GL1 = tf.reduce_sum(kernelonL1) / img_patch_norm

    # ssim_multiscale already calculates the dyalidic pyramid (with avg.pooling)
    msssims = tf.image.ssim_multiscale(img1=trueimg, img2=noisyimg, max_val=1)
    msssim = tf.reduce_sum(msssims)

    return alpha * (1 - msssim) + (1 - alpha) * GL1

print(loss_ssim_multiscale_GL1(img1, img2))



# plot kernel ------------------------------------------------------------------
import matplotlib.pyplot as plt
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    kernel = sess.run(
        tf.reshape(
            _fspecial_gauss(
                size=filter_size,
                sigma=filter_sigma),
            shape=(11, 11))
    )
plt.imshow(kernel, cmap='gray')
plt.show()




# ------------------------------------------------------------------------------
# ssim loss:
loss_ssim = 1-tf.image.ssim(img1=img1, img2=img2, max_val=1)
loss_ssim_multiscale= 1-tf.image.ssim_multiscale(img1=img1, img2=img2, max_val=1)