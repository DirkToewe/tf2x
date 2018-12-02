'''
Created on Dec 1, 2018

@author: Dirk Toewe
'''

import tensorflow as tf
from tf2x import tf2dot
from tempfile import mkdtemp

with tf.Graph().as_default() as graph:
  tf_a = tf.Variable([1,2,3], dtype=tf.float32, name='a')
  tf_b = tf.assign_add(tf_a, [4,5,6], name='b')
  with tf.control_dependencies([tf_b]):
    tf_c = tf.identity(tf_a, name='c')

  init_vars = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_vars)

    dot = tf2dot( graph, sess=sess )

    result_c = sess.run(tf_c)
    print('C[1]:', result_c)

    result_c = sess.run(tf_c)
    print('C[2]:', result_c)

tmpdir = mkdtemp()

dot.format = 'png'
dot.attr( root='b:0' )
dot.render(directory=tmpdir, view=True)
