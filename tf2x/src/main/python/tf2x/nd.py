'''
Utility methods for generating JavaScript code involving ND.JS from within Python.

Created on Dec 22, 2017

@author: Dirk Toewe
'''

import json, numpy as np
from base64 import b64encode

def arrayB64( ndarray: np.ndarray ) -> str:
  '''
  Returns a JavaScript nd.js nd.Array literal equivalent to the given
  numpy ndarray. The data is encoded in plain text.
  '''
  perLine = 256; assert perLine > 128

  ndarray = np.asarray(ndarray)

  dtype = ndarray.dtype.name
  shape = json.dumps(ndarray.shape)
  data = b64encode( ndarray.newbyteorder('<').tobytes() ).decode('UTF-8')

  if len(data) < perLine-64:
    return "nd.arrayFromB64('{}', {}, '{}')".format( dtype, shape, data )
  data = '\n  '.join(
    data[i*perLine : perLine*(i+1)]
    for i in range( 1 + ( len(data) - 1 ) // perLine )
  )
  return "nd.arrayFromB64('{}', {}, `\n  {}\n`)".format( dtype, shape, data )
  raise Exception('Not yet implemented.')

def array( ndarray: np.ndarray ) -> str:
  '''
  Returns an nd.js nd.Array literal equivalent to the given numpy ndarray.
  The data is encoded in Base64.
  '''
  def arrayStr( arr, indent='       ', prefix='[\n        ', suffix='\n      ]' ):
    '''
    Returns (as string) an (hopefully) pretty printed JavaScript representation of an ND-Array.
    '''
    if arr.ndim > 0:
      indent += ' '
      if arr.ndim == 1:
        prefix, suffix = '[', ']'
      infix = ', ' if arr.ndim == 1 else ',\n'+indent
      return prefix + infix.join( arrayStr(a,indent,'[',']') for a in arr) + suffix
    else:
      if isinstance(arr,np.integer):
        return repr( int(arr) )
      else:
        return repr( float(arr) )#np.array_str(arr, max_line_width=256, precision=1024, suppress_small=False)

  dtype = ndarray.dtype.name

  return "nd.array('{}', {})".format( dtype, arrayStr(ndarray) )
