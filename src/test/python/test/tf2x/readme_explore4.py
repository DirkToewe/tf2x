'''
Created on Dec 1, 2018

@author: Dirk Toewe
'''

import tensorflow as tf
from tf2x import tf2dot
from tempfile import mkdtemp

with tf.Graph().as_default() as graph:
  tf_a = tf.constant([1,2,3], dtype=tf.float32, name='a')
  tf_b = tf.constant([4,5,6], dtype=tf.float32, name='b')
  tf_c = tf.placeholder(shape=[], dtype=tf.bool, name='c')
  tf_d = tf.cond(tf_c, lambda: tf_a, lambda: tf_b, name='d')
  tf_e = tf.identity(tf_d, name='e')

  with tf.Session() as sess:
    dot = tf2dot( graph, sess=sess )

    result_e = sess.run(tf_d, feed_dict={tf_c: False})
    print('E(false):', result_e)

    result_e = sess.run(tf_d, feed_dict={tf_c: True })
    print('E(true):', result_e)

tmpdir = mkdtemp()

dot.format = 'png'
dot.attr( root='c:0' )
dot.render(directory=tmpdir, view=True)
