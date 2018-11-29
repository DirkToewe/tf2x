'''
Created on Oct 28, 2017

@author: Dirk Toewe
'''

import numpy as np, os, subprocess, tensorflow as tf
from tf2x import tf2graphml
from xml.dom import minidom
import xml.etree.ElementTree as xml
from tempfile import mkdtemp

def main():

#   loop = tf.while_loop(
#     cond = lambda i: i < tf.constant(100, name='Const100'),
#     body = lambda i: i + tf.constant(1,   name='Const1'  ),
#     loop_vars=[ tf.constant(0,name='INPUT') ],
#     parallel_iterations=1
#   )
#   y = tf.identity(loop, 'OUTPUT')

#   loop = tf.while_loop(
#     cond = lambda i: i*i < i * tf.constant(100, name='Const100'),
#     body = lambda i: i + tf.constant(1,   name='Const1'  ),
#     loop_vars=[ tf.constant(0,name='INPUT') ],
#     parallel_iterations=1
#   )
#   y = tf.identity(loop, 'OUTPUT')


#   A = tf.Variable(12, name='A')
#   B = tf.Variable(15, name='B')
#   C = tf.constant(17, name='C')
#   y = tf.cond(A < 12, lambda: B, lambda: C)

  INPUT = tf.constant(
#     np.column_stack([
      np.arange(1000),
#       np.arange(1000,2000)
#     ]),
    name='INPUT'
  )
  print(INPUT.shape)
  OUTPUT = tf.map_fn(lambda x: tf.Print(x*1337, [x]), INPUT, parallel_iterations=1)
  OUTPUT = tf.identity(OUTPUT, name='OUTPUT')

  init_vars = tf.global_variables_initializer()

  cfg = tf.ConfigProto(
    device_count = {'GPU': 0}
  )

  with tf.Session( config=cfg ) as sess:

    sess.run(init_vars)

#     graph_doc = graph2graphml( tf.get_default_graph() )
    graph_doc = tf2graphml(OUTPUT)
    tmpdir = mkdtemp()
    filepath = os.path.join(tmpdir, 'test.graphml')
    print(filepath)

    graph_doc = minidom.parseString(
      xml.tostring(graph_doc.getroot(), 'unicode')
    ).toprettyxml(indent="  ")

    with open(filepath, "w") as f:
      f.write(graph_doc)

#     graph_doc.write(filepath, xml_declaration=True, encoding='UTF-8')

    subprocess.Popen(
      ['java', '-jar', 'yed.jar', filepath],
      cwd='/opt/yEd/',
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE
    )


if '__main__' == __name__:
  main()

