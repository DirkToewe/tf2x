'''
Created on Dec 18, 2017

@author: Dirk Toewe
'''

import tensorflow as tf
from tf2x.inspect import trainable_vars

import numpy as np
from numbers import Number


if __name__ == '__main__':
  x = tf.Variable(1, name='X', trainable=True)
  y = tf.Variable(2, name='Y', trainable=False)
  z = tf.Variable(3, name='Z', trainable=True)
  vars = trainable_vars(3*x + 4*(y+2*z) )
  print('\n'.join( map(str,vars) ) )


