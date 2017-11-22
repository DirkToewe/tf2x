'''
Deploys the convolutional neural network (CNN) trained in mnist_train.py in form of
a standalone HTML-page.

Created on Sep 9, 2017

@author: Dirk Toewe
'''

import os, tensorflow as tf
from tempfile import mkdtemp
import webbrowser

from pkg_resources import resource_string

from mnist_train import Model
import numpy as np, tf2x
from tf2x.tf2js import tensor2js


_nd_js = resource_string(tf2x.__name__, 'nd.js').decode('utf-8')
_gui_template = resource_string(__name__, 'mnist_gui.html.template').decode('utf-8')


def main():

  w,h = 28,28

  model = Model(w,h, deploy=True)

  init_vars = tf.global_variables_initializer()
  saver = tf.train.Saver( keep_checkpoint_every_n_hours=1 )

  tmp_dir = mkdtemp()

  with tf.Session() as sess:
    sess.run(init_vars)
    model_path = os.path.expanduser('~/Pictures/MNIST/summary/model.ckpt-2260')
    saver.restore(sess, model_path)
    model_js = tensor2js(model.out_prediction, sess=sess)

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

  html_gui = _gui_template.format(
    ND =_nd_js,
    MODEL = model_js
      .replace('{','\\{')
      .replace('}','\\}')
      .replace('`','\\`')
  )

  raw_path = os.path.join(tmp_dir,'mnist_gui.html')

  with open(raw_path, 'w') as raw:
    raw.write(html_gui)

  print('tmp_dir: ' + tmp_dir)
  print()
  webbrowser.open('file://'+raw_path)


if '__main__' == __name__:
  main()