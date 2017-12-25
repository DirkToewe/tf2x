'''
Created on Dec 25, 2017

@author: Dirk Toewe
'''

import numpy as np
from numbers import Number


def array2str( array: np.ndarray, val2str=None ) -> str:
  '''
  Returns a pretty-printed string representation of a NumPy array.
  If the string is to large, ellipses are inserted and only a fraction
  of the string is displayed in the string.

  Parameters
  ----------
  array: np.ndarray
    
  val2str: dtype => str
    The function used to convert the entry values to string.
  ''' 
  array = np.asarray(array)
  idx = np.zeros(array.ndim, dtype=np.int)

  nSqueeze = array.ndim
  while nSqueeze > 0 and array.shape[nSqueeze-1] == 1:
    nSqueeze -= 1

  if   nSqueeze <= 2: MAX_LEN = 10; FIRST= 4; LAST= 4
  elif nSqueeze <= 3: MAX_LEN = 5;  FIRST= 2; LAST= 2
  else              : MAX_LEN = 3;  FIRST= 1; LAST= 1

  if None is val2str:
    def _val2str( value ):
      if isinstance(value, Number):
        return '{:.6g}'.format(value)
      else:
        return str(value)

    def printedEntries( dim ):
      if dim == array.ndim:
        yield _val2str( array[tuple(idx)] )
      else:
        l = array.shape[dim]
        if l > MAX_LEN:
          for idx[dim] in range( 0,  FIRST): yield from printedEntries(dim+1)
          for idx[dim] in range(l-1-LAST,l): yield from printedEntries(dim+1)
        else:
          for idx[dim] in range( 0,      l): yield from printedEntries(dim+1)

    lSize=0
    rSize=0
 
    # find largest common size
    for s in printedEntries(0):
      spl = s.split('.')
      assert len(spl) <= 2
      if len(spl) == 2:
        rSize = max(rSize, len(spl[1]) )
      lSize = max(lSize, len(spl[0]) )
 
    def val2str( val ):
      s = _val2str(val)
      spl = s.split('.')
      assert len(spl) <= 2
      rs = 0
      if len(spl) == 2:
        rs = len(spl[1])
      ls = len(spl[0])
      return ' '*(lSize-ls)  +  s  +  ' '*(rSize-rs-len(spl)+2)

  if array.ndim == 0:
    return val2str( float(array) )

 
  def array2str( dim, indent ) -> str:
    assert dim <= array.ndim
    if dim == array.ndim:
      yield val2str( array[tuple(idx)] )
    else:
      l = array.shape[dim]
      sep = ',\n '+indent if dim < nSqueeze-2 else ', '

      def nextSlice():
        result = sep.join( array2str(dim+1, indent+' ') )
        if dim < array.ndim-1:
          result = '['+result+']'
        return result

      if l > MAX_LEN:
        for idx[dim] in range( 0,  FIRST): yield nextSlice()
        yield '...%d more...' % (l-LAST-FIRST)
        for idx[dim] in range(l-1-LAST,l): yield nextSlice()
      else:
        for idx[dim] in range( 0,      l): yield nextSlice() 

  sep = ', ' if nSqueeze <= 1 else ',\n '
  return '[' + sep.join( array2str(0,' ') ) + ']'


