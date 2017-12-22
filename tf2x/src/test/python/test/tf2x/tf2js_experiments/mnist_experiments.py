'''
Deploys the convolutional neural network (CNN) trained in mnist_train.py in form of
a standalone HTML-page.

Created on Sep 9, 2017

@author: Dirk Toewe
'''

import os, subprocess, tensorflow as tf
from tempfile import mkdtemp

from pkg_resources import resource_string
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

import numpy as np, tf2x
from test.tf2x.tf2js_experiments.MNIST_Model import MNIST_Model
from test.tf2x.tf2js_experiments.mnist_train import PROJECT_DIR
from tf2x import tensor2js#, tf2dot
from datetime import datetime
from tf2x import nd


_nd_js = resource_string(tf2x.__name__, 'nd.js').decode('utf-8')
_script_template = '''
'use strict'

console.log("loading nd.js...")

{ND}

console.log("loading model...")

const Model = {MODEL}

console.log("instantiating model...")

const model = new Model()

console.log("parsing input...")

const input = {INPUT}

console.log("executing model...")


let indent = ''

for( const [name,op] of Object.entries(model.ops) )
  model.ops[name] = inputs => {{
    console.log(indent+name)
    console.time(indent+name)
    indent += '│'
    const result = op(inputs)
    indent = indent.slice(0,-1)
    console.timeEnd(indent+name)
    return result
  }}


for( let i=1; i-- > 0; )
{{
  let output = model({{ 'in/images:0': input }})
  output = output.map( x => x.toExponential(8) )
  console.log('')
  console.log( output.toString() )
}}
'''

def main():

  w,h = 28,28

  mnist = mnist_data.read_data_sets( os.path.join(PROJECT_DIR, 'data'), one_hot=True)

#   in_images_train     = mnist.train     .images.reshape(-1,h,w,1)
#   in_images_validation= mnist.validation.images.reshape(-1,h,w,1)
  in_images_test      = mnist.test      .images.reshape(-1,h,w,1) # <- used to estimate the model error during training
  in_images_test = in_images_test[:100]
    
#   in_labels_train     = mnist.train     .labels
#   in_labels_validation= mnist.validation.labels
#   in_labels_test      = mnist.test      .labels # <- used to estimate the model error during training

  model = MNIST_Model(w,h, deploy=True)

  init_vars = tf.global_variables_initializer()
  saver = tf.train.Saver( keep_checkpoint_every_n_hours=1 )

  tmp_dir = mkdtemp()

  with tf.Session() as sess:

    sess.run(init_vars)
    model_path = os.path.join(PROJECT_DIR, 'summary/model.ckpt-2100')
    saver.restore(sess, model_path)

    t0 = datetime.now()
    result = sess.run( model.out_prediction, feed_dict={model.in_images: in_images_test} )
    dt = datetime.now() - t0

#     dot = tf2dot(model.out_prediction, sess=sess)
#     dot.format = 'svg'
#     dot.render( os.path.join(tmp_dir,'graph.gv'), view=True )

    model_js = tensor2js(model.out_prediction, sess=sess)

  script = _script_template.format(
    ND =_nd_js,
    MODEL = model_js,
    INPUT = nd.arrayB64(in_images_test)
  )

  raw_path = os.path.join(tmp_dir,'raw.js')
#   opt_path = os.path.join(dir_path,'opt.js')

  with open(raw_path, 'w') as raw:
    raw.write(script)

#   subprocess.Popen([
#     'npx', 'google-closure-compiler',
#     '--js="%s"' % raw_path,
#     '--js_output_file="%s"' % opt_path
#   ]).wait()

  print('tmp_dir: ' + tmp_dir)
  print()
  proc = subprocess.Popen(['node', '--max_old_space_size=12288', raw_path], stderr=subprocess.STDOUT)
  proc.wait()
  print( np.array2string( result, separator=', ', max_line_width=256 ) )
  print()
  print(dt)



if '__main__' == __name__:
  main()