'''
Created on Dec 14, 2017

@author: Dirk Toewe
'''
from itertools import chain, product 
import json, os, subprocess, tensorflow as tf
from tempfile import mkdtemp

from pkg_resources import resource_string

import numpy as np, tf2x
from tf2x import tensor2js


_nd_js = resource_string(tf2x.__name__, 'nd.js').decode('utf-8')
_script_template = '''
'use strict'

{ND}

const Model = {MODEL}

const model = new Model()

let output = model({{}})
console.log( output.slice('...',0).toString(Infinity) )
'''


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


def test_offset( padding_N ):
  padding,N = padding_N
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#   print( 'N=%s; PAD="%s"' % (N,padding) )

  tmp_dir = mkdtemp()
  F_MAX = N if padding == 'VALID' else N+3

  for S,F in product( np.ndindex(*N), np.ndindex(*F_MAX) ):
    S = np.array(S) + 1
    F = np.array(F) + 1
#     print( '  N=%s; S=%s; F=%s; PAD="%s"' % (N,S,F,padding) )

    input_data = np.reshape([
      10*i+j + 11
      for i in range(N[0])
      for j in range(N[1])
    ], [1,*N,1])
  
    with tf.Graph().as_default():
      input = tf.constant(
        input_data,
        dtype=tf.float32
      )
  
      filter=np.zeros([*F,1,1])
      filter[
        (F[0]-1) >> 1,
        (F[1]-1) >> 1
      ] = 1

      output = tf.nn.conv2d(input, filter, [1,*S,1], padding, use_cudnn_on_gpu=False)
    
      with tf.Session() as sess:
        model_js = tensor2js(output, sess=sess)
        out_tf = sess.run(output)[...,0]
    
    script = _script_template.format(
      ND =_nd_js,
      MODEL = model_js
    )
  
    script_path = os.path.join(tmp_dir,'script.js')
  
    with open(script_path, 'w') as raw:
      raw.write(script)
  
    proc  = subprocess.Popen(['node', '--max_old_space_size=4096', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out_js,err_js = proc.communicate()

    if 0 < len(err_js):
      raise AssertionError('Node script execution failed: '+err_js)

    out_js = np.array( json.loads(out_js) )
    assert out_tf.shape == out_js.shape

    if not np.allclose(out_tf, out_js):
      print(out_js)
      print(out_tf)
    assert np.allclose(out_tf, out_js), (
      "\nINPUT(%s):\n%s\nFILTER(%s):\n%s\nSTRIDES: %s\n\n\n%s\n!=\n%s" % (
        input_data.shape,
        input_data[...,0],
        filter.shape,
        filter[...,0],
        S,
        out_tf,
        out_js
      )
    )


def test_dilation( padding_N ):
  padding,N = padding_N
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#   print( 'N=%s; PAD="%s"' % (N,padding) )

  tmp_dir = mkdtemp()
  F_MAX = N if padding == 'VALID' else N+3

  for S,F in product( np.ndindex(*N), np.ndindex(*F_MAX) ):
    S = np.array(S) + 1
    F = np.array(F) + 1

    D_MAX = np.maximum(1,N // F) # np.maximum(1,N-F+1) if padding == 'SAME' else N+3

    for D in np.ndindex(*D_MAX):
      D = np.array(D) + 1

#       print( '  N=%s; S=%s; F=%s; D=%s; PAD="%s"' % (N,S,F,D,padding) )
  
#       input_data = np.reshape([
#         10*i+j + 11
#         for i in range(N[0])
#         for j in range(N[1])
#       ], [1,*N,1])
      input_data = np.random.rand(1,*N,1)*8 - 4
    
      with tf.Graph().as_default():
        input = tf.constant(
          input_data,
          dtype=tf.float32
        )
    
        filter=np.random.rand(*F,1,1)*2 - 1
#         filter[
#           (F[0]-1) >> 1,
#           (F[1]-1) >> 1
#         ] = 1
  
        output = tf.nn.conv2d(input, filter, strides=[1,*S,1], dilations=[1,*D,1], padding=padding, use_cudnn_on_gpu=False) 
      
        with tf.Session() as sess:
          model_js = tensor2js(output, sess=sess)
          out_tf = sess.run(output)[...,0]
      
      script = _script_template.format(
        ND =_nd_js,
        MODEL = model_js
      )
    
      script_path = os.path.join(tmp_dir,'script.js')
    
      with open(script_path, 'w') as raw:
        raw.write(script)
    
      proc  = subprocess.Popen(['node', '--max_old_space_size=4096', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
      out_js,err_js = proc.communicate()
  
      if 0 < len(err_js):
        raise AssertionError('Node script execution failed: '+err_js)
  
      out_js = np.array( json.loads(out_js) )
      assert out_tf.shape == out_js.shape
  
#       if not np.allclose(out_tf, out_js, atol=1e-4, rtol=1e-4):
#         print(out_tf)
#         print(out_js)
      assert np.allclose(out_tf, out_js, atol=1e-4, rtol=1e-4), (
        "\nINPUT(%s):\n%s\nFILTER(%s):\n%s\nSTRIDES: %s\n\n\n%s\n!=\n%s" % (
          input_data.shape,
          input_data[...,0],
          filter.shape,
          filter[...,0],
          S,
          out_tf,
          out_js
        )
      )


def run_tests():
  '''
  Generates the tests.
  '''
  import multiprocessing
  pool = multiprocessing.Pool()

  try:
    print('START')

    inputs = [
      (padding, np.array(N) + 1)
      for N in np.ndindex(8,8)
      for padding in ['SAME','VALID']
    ]

#     for padding_N in inputs:
#       test_dilation(padding_N)
    
    jobs = chain.from_iterable([
      pool.imap_unordered(f,inputs)
      for f in [test_dilation,test_offset]
    ])
    pool.close()
    for _ in jobs:
      pass
  finally:
    pool.terminate()


if '__main__' == __name__:
  run_tests()
