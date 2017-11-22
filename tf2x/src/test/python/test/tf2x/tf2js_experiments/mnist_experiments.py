'''
Deploys the convolutional neural network (CNN) trained in mnist_train.py in form of
a standalone HTML-page.

Created on Sep 9, 2017

@author: Dirk Toewe
'''

import json, os, subprocess, tensorflow as tf
from tempfile import NamedTemporaryFile, mkdtemp

from pkg_resources import resource_string
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from mnist_train import Model
import numpy as np, tf2x
from tf2x.tf2js import tensor2js
from tf2x.tf2dot import tf2dot


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

const input = nd.array('float32',{INPUT})

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

  mnist = mnist_data.read_data_sets( os.path.expanduser('~/Pictures/MNIST/data'), one_hot=True)

#   in_images_train     = mnist.train     .images.reshape(-1,h,w,1)
#   in_images_validation= mnist.validation.images.reshape(-1,h,w,1)
  in_images_test      = mnist.test      .images.reshape(-1,h,w,1) # <- used to estimate the model error during training
  in_images_test = in_images_test[:1]
    
#   in_labels_train     = mnist.train     .labels
#   in_labels_validation= mnist.validation.labels
#   in_labels_test      = mnist.test      .labels # <- used to estimate the model error during training

  model = Model(w,h, deploy=True)

  init_vars = tf.global_variables_initializer()
  saver = tf.train.Saver( keep_checkpoint_every_n_hours=1 )

  tmp_dir = mkdtemp()

  with tf.Session() as sess:

    sess.run(init_vars)
    model_path = os.path.expanduser('~/Pictures/MNIST/summary/model.ckpt-2260')
    saver.restore(sess, model_path)

    result = sess.run( model.out_prediction, feed_dict={model.in_images: in_images_test} )

#     dot = tf2dot(model.out_prediction, sess=sess)
#     dot.format = 'svg'
#     dot.render( os.path.join(tmp_dir,'graph.gv'), view=True )

    model_js = tensor2js(model.out_prediction, sess=sess)

    print( np.array2string( result, separator=', ', max_line_width=256 ) )
    print()

  def arrayStr( arr, indent='   ', prefix='[\n    ', suffix='\n  ]' ):
    '''
    Returns (as string) an (hopefully) pretty printed JavaScript representation of an ND-Array.
    '''
    if arr.ndim > 0:
      indent += ' '
      if arr.ndim == 1:
        prefix,suffix = '[', ']'
      infix = ', ' if arr.ndim == 1 else ',\n'+indent
      return prefix + infix.join( arrayStr(a,indent,'[',']') for a in arr) + suffix
    else:
      if isinstance(arr,np.integer):
        return repr( int(arr) )
      else:
        return repr( float(arr) )#np.array_str(arr, max_line_width=256, precision=1024, suppress_small=False)

  script = _script_template.format(
    ND =_nd_js,
    MODEL = model_js,
    INPUT = arrayStr(in_images_test)
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


if '__main__' == __name__:
  main()