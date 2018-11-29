'''
Created on Sep 14, 2017

@author: Dirk Toewe
'''

import tensorflow as tf
import numpy as np
import numpy.linalg as la
from math import radians as deg2rad

class ImageDistorter(object):
  '''
  The ImageDistorter takes Tensorflow images as input an returns an optically
  distorted and noisy transformation of said image. Using the correct settings,
  noise and distortion should be small enough to not impair the human readability
  of the images. Just like a human, the machine should be able to handle such
  distortions.

  For each individual image data set, the ImageDistorter parameters should
  be tweaked and inspected manually.

  The main goal of the ImageDistorter is to prevent overfitting. A single
  pixel of 256 shades of grey may be enough to distinguish up to 256 images.
  Four pixels may be enough to distinguish up to over 4 billion images. Clearly,
  a Machine Learning algorithm should look at more than just 4 pixels. Adding
  artificial distortions and noise during training ensures that individual
  pixels almost never have the same value, effectively forcing the classifier
  to look at the entire image.

  Another way to look at the ImageDistorter is as a way to artificially generate
  more training data from the few images that are available for training.

  Parameters
  ----------
  salt: float
    The probability of salt noise (pixels set to white)
  pepper: float
    The probability of peppeer nosie (pixels set to black)
  noise: float
    The standard deviation of the gaussian noise added to each pixel.
  swirl: float
    The magnitude of the angular, vortex-like distortion of the images.
  warp: float
    The magnitude of linear distorting in x- and y-direction of the images (shorten/stretch locally).
  rotate: float
    The magnitude of random rotation of the image.
  move: (int,int)
    The vertical and horizontal random movement. The image size remains unchanged. Pixels moved away are replaced by 0.
  '''

  def __init__(self, *, salt=0.001, pepper=0.001, noise=0.1, swirl=6, warp=0.75, rotate=8, move=(2,2) ):
    self._salt = salt
    self._pepper = pepper
    self._noise = noise
    self._swirl = swirl
    self._warp = warp
    self._rotate = rotate
    self._move = move


  def swirl(self, image):
    h,w,_ = image.shape
   
    x,y = np.meshgrid(
      np.linspace(-1.0, +1.0, h),
      np.linspace(-1.0, +1.0, w)
    )
    xy = np.dstack([x,y])
   
    angles = np.arctan2(x,y)
    dist = la.norm(xy,axis=-1)
    mask = tf.constant(dist <= 1, dtype=tf.float32)
     
    n = 5
   
    def deltaAngle():
   
      x = 1-dist
      x = x*x
      t,T = 1, x
      result = tf.random_uniform([], -1, +1) * T
    
      for _ in range(n-1):
        t,T = T, 2*x * T - t
        t,T = T, 2*x * T - t
        result += tf.random_uniform([], -1, +1) * T
    
      return result
   
    # use random combination of chebychev polynomials since they are bounded in [0,1]
    # TODO use tschebychev recursion formula
    deltaAngle = deltaAngle() * deg2rad(self._swirl)
   
    angles = angles + mask * deltaAngle
     
    x = dist * tf.cos(angles)
    y = dist * tf.sin(angles)
   
    x = (x+1) / 2 * tf.cast(h-1, tf.float32)
    y = (y+1) / 2 * tf.cast(w-1, tf.float32)
    xy = tf.stack([x,y], axis=2)
    xy = tf.cast( tf.round(xy), tf.int32 )
   
    return tf.gather_nd(image, xy)


  def warp_v(self, image):
    h,_,_ = image.shape
    yEnd = tf.cast(h-1, tf.float32)
    y = tf.lin_space(0.0, 1.0, h)
    
    n = 16
    
    def deltaY(): # <- use random combination of Chebychev polynomials since they are bounded in [0,1]
       
      t,T = 1, y
      result = tf.random_uniform([], -1, +1) * T
   
      for _ in range(n-1):
        t,T = T, 2*y * T - t
        t,T = T, 2*y * T - t
        result += tf.random_uniform([], -1, +1) * T
   
      return result
   
    deltaY = deltaY()
    deltaY *= tf.random_uniform([], 0, self._warp) / ( 1e-128 + tf.reduce_max( tf.abs(deltaY) ) )
    y = y*yEnd + tf.cumsum(deltaY)
   
    y *= yEnd / y[-1]
    y  = tf.cast( tf.round(y), tf.int32 )
    return tf.gather(image, y)


  def warp_h(self, image):
    image = tf.transpose(image, [1, 0, 2]); image = self.warp_v(image)
    image = tf.transpose(image, [1, 0, 2])
    return image


  def __call__(self, image):
    '''
    Returns a Tensorflow computation pipeline for the distortion
    of the given input image.

    Parameters
    ----------
    image: tf.Tensor[height,width,channels]

    Returns
    -------
    distorted: tf.Tensor[height,width,channels]
    '''
    h,w,_ = image.shape
    h = tf.cast(h, tf.int32)
    w = tf.cast(w, tf.int32)

    random_angle = tf.random_uniform([], minval=deg2rad(-self._rotate), maxval=deg2rad(+self._rotate), dtype=tf.float32)
    random_noise = tf.random_normal([h,w,1], 0.0, self._noise)
    random_salt_n_pepa = tf.random_uniform([h,w,1], 0.0, 1.0)
    random_salt = tf.cast( random_salt_n_pepa >= 1-self._salt, tf.float32)
    random_pepa = tf.cast( random_salt_n_pepa <  self._pepper, tf.float32)
    random_dv = tf.random_uniform( [], -self._move[0], +self._move[0]+1, dtype=tf.int32)
    random_dh = tf.random_uniform( [], -self._move[1], +self._move[1]+1, dtype=tf.int32)
# 
#     image = tf.image.per_image_standardization(image)
#
    image = tf.contrib.image.rotate(image, random_angle)

    image = image[
      tf.maximum(0,random_dv) : h+tf.minimum(0,random_dv),
      tf.maximum(0,random_dh) : w+tf.minimum(0,random_dh)
    ]
    image = tf.pad(image, [
      (tf.maximum(0,-random_dv), tf.maximum(0,random_dv)),
      (tf.maximum(0,-random_dh), tf.maximum(0,random_dh)),
      (0,0)
    ])

    image = tf.image.random_brightness(image, self._noise)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image += random_noise
    image = (1-random_salt)*image + random_salt
    image = (1-random_pepa)*image

    image = self.swirl(image)
    image = self.warp_v(image)
    image = self.warp_h(image)

    return image
