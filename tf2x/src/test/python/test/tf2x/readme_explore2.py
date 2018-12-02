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

  init_vars = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    dot = tf2dot( graph, sess=sess )

    sess.run(init_vars)

    result_a = sess.run(tf_a)
    print('A:')
    print(result_a)

tmpdir = mkdtemp()

dot.format = 'png'
dot.attr( root='b:0' )
dot.render(directory=tmpdir, view=True)
