'''
Created on Dec 14, 2017

@author: Dirk Toewe
'''
from itertools import product
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



def main():
  
    with tf.Graph().as_default():

      output = tf.constant([
        [1,0,0],
        [0,1,0],
        [0,0,1]
      ])
    
      with tf.Session() as sess:
        model_js = tensor2js(output, sess=sess)
        out_tf = sess.run(output)[...,0].astype(np.int)
    
    script = _script_template.format(
      ND =_nd_js,
      MODEL = model_js
    )

    tmp_dir = mkdtemp()
    print(tmp_dir)
    script_path = os.path.join(tmp_dir,'script.js')
  
    with open(script_path, 'w') as raw:
      raw.write(script)
  
    proc  = subprocess.Popen(['node', '--max_old_space_size=4096', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out_js,err_js = proc.communicate()

    if 0 < len(err_js):
      raise AssertionError('Node script execution failed: '+err_js)

    print(out_js)
    out_js = np.array( json.loads(out_js) )
    print(out_js)



if '__main__' == __name__:
  main()


