'''
Created on Sep 8, 2017

@author: Dirk Toewe
'''

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import tensorflow as tf
import matplotlib.pyplot as plt
from tf2x.tf2js_experiments import ImageDistorter

def main():
  mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=True)

  in_images = tf.placeholder(tf.float32, [None,28,28,1], 'in_images')

  out_images = tf.map_fn(ImageDistorter(), in_images)

  feed_in_images = mnist.validation.images.reshape(-1,28,28,1)

#   nRows, nCols = 1,2
  nRows, nCols = 6,9
  fig1 = plt.figure( tight_layout=True )

  with tf.Session() as sess:
    for row in range(nRows):
      for col in range(nCols):
  
        print(row,col)

        feed_dict = {
          'in_images:0': feed_in_images
        }
  
        res_out_image = sess.run(out_images, feed_dict=feed_dict)

        sub1 = fig1.add_subplot(nRows,nCols, 1 + nCols*row + col)
        sub1.set_xticklabels([])
        sub1.set_yticklabels([])
        sub1.axis('off')
        sub1.imshow(res_out_image[512,:,:,0], cmap='viridis')
  
  fig1.subplots_adjust(wspace=None, hspace=None)
  plt.show()
  

if __name__ == '__main__':
  main()