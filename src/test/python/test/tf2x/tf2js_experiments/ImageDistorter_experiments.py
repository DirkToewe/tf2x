'''
Created on Sep 8, 2017

@author: Dirk Toewe
'''

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from test.tf2x.tf2js_experiments.ImageDistorter import ImageDistorter

def main():
  mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=True)

  in_images = tf.placeholder(tf.float32, [None,28,28,1], 'in_images')

  out_images = tf.map_fn(ImageDistorter(), in_images)

  mnist_images = mnist.validation.images.reshape(-1,28,28,1)

#   nRows, nCols = 1,2
  nRows, nCols = 6,9

  with tf.Session() as sess:
    for img in mnist_images:
      img = np.stack( [img]*(nRows*nCols) )
      img = sess.run(out_images, feed_dict={in_images: img})

      fig1 = plt.figure( tight_layout=True )

      for i,im in enumerate(img,1):
        sub1 = fig1.add_subplot(nRows,nCols, i)
        sub1.set_xticklabels([])
        sub1.set_yticklabels([])
        sub1.axis('off')
        sub1.imshow(im[...,0], cmap='magma')
  
      fig1.subplots_adjust(wspace=None, hspace=None)
      plt.show()
  

if __name__ == '__main__':
  main()
