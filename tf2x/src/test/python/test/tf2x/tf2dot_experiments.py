'''
Created on Nov 10, 2017

@author: Dirk Toewe
'''

import numpy as np, tensorflow as tf
from tf2x import tf2dot
from tempfile import mkdtemp

def main():

#   loop = tf.while_loop(
#     cond = lambda i: i < tf.Variable(100, name='Const100'),
# #     cond = lambda i: i < tf.constant(5, name='Const5') + tf.constant(100, name='Const100'),
#     body = lambda i: i + tf.constant(1,   name='Const1'  ),
#     loop_vars=[ tf.placeholder(tf.int32, name='INPUT') ],
#     parallel_iterations=1
#   )
#  
#   y = tf.identity(loop, 'OUTPUT')

#   x = tf.Variable(0, name='Const0')
#   y = tf.cond(
#     x < tf.constant(1, name='Const1'),
#     lambda: tf.identity( tf.constant(2, name='Const2') ),
#     lambda: tf.Variable(3, name='Const3')
#   )

#   INPUT = tf.constant(
# #     np.column_stack([
#       np.arange(4),
# #       np.arange(1000,2000)
# #     ]),
#     name='INPUT'
#   )
#   OUTPUT = tf.map_fn(lambda x: x*1337, INPUT, parallel_iterations=1)
# #   OUTPUT = tf.map_fn(lambda x: tf.Print(x*1337, [x]), INPUT, parallel_iterations=1)
#   OUTPUT = tf.identity(OUTPUT, name='OUTPUT')

#   OUTPUT = tf.TensorArray(tf.float64, size=4)

  INPUT = tf.constant([1,2,3], name='INPUT')
  MAP_FN = tf.map_fn( lambda i: tf.multiply(i,i, name='I_TIMES_I'), INPUT, name='MAP_FN' )
  OUTPUT = tf.identity(MAP_FN, name='OUTPUT')

  with tf.Session() as sess:

#     sess.run( tf.global_variables_initializer() )
#     print( sess.run(y, feed_dict = {'INPUT:0': 0}) )

    print( sess.run(OUTPUT) )

    dot = tf2dot(OUTPUT, sess=sess)
    dot.format = 'pdf'
    dot.attr( root='INPUT' )
#     dot.attr( splines='true')
#     dot.attr( rank='same' )
#     dot.attr( ranksep='0.01', nodesep='0.01' )
#     dot.attr( rankdir='LR' )
#     dot.attr(splines='ortho', layout='circo')

    tmpdir = mkdtemp()
    help(dot.render)
    dot.render(directory=tmpdir, view=True)


if __name__ == '__main__':
  main()

