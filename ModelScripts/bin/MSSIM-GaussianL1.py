import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

imgs = np.load('C:/Users/timru/Documents/CODE/deepMRI1/Data/P1_X.npy')
img1 = tf.reshape(imgs[0], (1,256,256,1))
img2 = tf.reshape(imgs[1], (1,256,256,1))

# (gaussian kernel)
# source: https://gist.github.com/charliememory/91a62ff0a102bc0bfe715711fd5c538d
# :size: size*size is grid at which. should be uneven, so that image centroid is at gaussian mean
# :sigma: standard deviation of Gaussian.
size = 11
sigma = 4

xaxis, yaxis = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
xaxis = np.expand_dims(xaxis, axis=-1)
xaxis = np.expand_dims(xaxis, axis=-1)
yaxis = np.expand_dims(yaxis, axis=-1)
yaxis = np.expand_dims(yaxis, axis=-1)

x = tf.constant(xaxis, dtype=tf.float32)
y = tf.constant(yaxis, dtype=tf.float32)
g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
g = g / tf.reduce_sum(g) # normalizing on grid

# to calculate kernel only once on runtime
g = tf.Variable(tf.reshape(g, (size, size, 1)))
g.size = size


def gaus_l1(img1, img2, stride, alpha=0.3, kernel=g):
    ''':stride: shift of the centroid in both width and height'''
    nimg = tf.extract_image_patches(
        images=img1,
        ksizes=[1, g.size, g.size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')

    timg = tf.extract_image_patches(
        images=img2,
        ksizes=[1, g.size, g.size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')

    x = tf.reshape(nimg, [-1, g.size, g.size, 1])
    y = tf.reshape(timg, [-1, g.size, g.size, 1])
    z = x - y

    # where g is gaussian kernel
    L1 = tf.reduce_sum(tf.map_fn(fn=lambda p: p * g, elems=z))

    #ms-ssim on patches (parallel traversel)
    # TODO the msssim parallel traversel on patches does not work.

    c = tf.stack([x, y], axis=1)
    msssims = tf.map_fn(
        fn=lambda x: tf.image.ssim_multiscale(x[0], x[1], max_val=1),
        elems=c,
        dtype=tf.float32
    )
         #msssim = tf.reduce_sum(msssims)

         #loss = (1 - alpha) * msssim + alpha * L1

    return msssims

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    c =sess.run(gaus_l1(img1=img1, img2=img2, stride = 4, alpha = 0.3))

# (LOSS: weighted MS-SSIM + G*L1) ----------------------------------------------
# :alpha: scalar value, weight: alpha * l1 + (1-alpha) * ms-ssim
msssims = tf.image.ssim_multiscale(img1=img1, img2=img2, max_val=1),
msssim = tf.reduce_sum(msssims)
alpha = 0.3
loss = (1 - alpha) * msssim + alpha * gaus_l1(img1=img1, img2=img2, stride = 4, alpha = 0.3)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(gaus_l1(img1=img1, img2=img2, stride = 4, alpha = 0.3)))
    kernel = sess.run(g)


plt.imshow(np.reshape(kernel, (size,size)), cmap='gray')
plt.show()




