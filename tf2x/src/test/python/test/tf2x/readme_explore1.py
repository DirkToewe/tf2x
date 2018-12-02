'''
Created on Dec 1, 2018

@author: Dirk Toewe
'''

import tensorflow as tf
from tf2x import tf2dot
from tempfile import mkdtemp

tf_a = tf.constant([1,2,3], dtype=tf.float32, name='a')
tf_b = tf.placeholder(      dtype=tf.float32, name='b')
tf_c = tf.add(tf_a,tf_b, name='c')

with tf.Session() as sess:
  dot = tf2dot( tf_c, sess=sess )

tmpdir = mkdtemp()

dot.format = 'png'
dot.render(directory=tmpdir, view=True)
