'''
This example demonstrates how TF2X can be used to convert a Tensorflow
computation graph into JS code and how to execute that code on NodeJS.

Created on Nov 30, 2019

@author: Dirk Toewe
'''
import numpy as np, os, subprocess, tensorflow as tf, tf2x
from pkg_resources import resource_string
from tempfile import mkdtemp
from tf2x import nd, tensor2js

ndjs_code = resource_string(tf2x.__name__, 'nd.js').decode('utf-8')

js_code_template = '''
{ND}
{Model}
const model = new MyModel()
const a_data = {a_data}
let output = model({{ 'a:0': a_data }})
console.log('\\nOutput JS:')
console.log( output.toString() )
'''

tf_a = tf.placeholder(name='a', shape=[3], dtype=tf.float32)
tf_b = tf.constant(10, name='b', dtype=tf.float32)
tf_c = tf_a + tf_b

with tf.Session() as sess:
  tf_out = sess.run( tf_c, feed_dict={ tf_a: [1,2,3] } )
  print('Output TF:')
  print( np.array2string( tf_out, separator=', ', max_line_width=256 ) )

  js_code = js_code_template.format(
    ND = ndjs_code,
    Model = tensor2js(tf_c, sess=sess, model_name='MyModel'),
    a_data = nd.arrayB64( np.array([1,2,3], dtype=np.float32) )
  )

js_dir = mkdtemp()
js_file = os.path.join(js_dir, 'main.js')

with open(js_file, 'w') as fout:
  fout.write(js_code)

proc = subprocess.Popen(['node', 'main.js'], stderr=subprocess.STDOUT, cwd=js_dir)
proc.wait()
