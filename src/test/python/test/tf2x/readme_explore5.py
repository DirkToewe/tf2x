'''
Created on Dec 1, 2018

@author: Dirk Toewe
'''

import tensorflow as tf
from tf2x import tf2dot
from tempfile import mkdtemp

tf_loop = tf.while_loop(
  cond = lambda i: i < tf.constant(16, name='iMax'),
  body = lambda i: i+1,
  loop_vars = (tf.constant(0, name='i0'),)
)
tf_out = tf.identity(tf_loop, name='out')

with tf.Session() as sess:
  dot = tf2dot( tf_out, sess=sess )

tmpdir = mkdtemp()
 
dot.format = 'png'
dot.attr( root='i0' )
dot.attr( newrank='true' )
dot.render(directory=tmpdir, view=True)
