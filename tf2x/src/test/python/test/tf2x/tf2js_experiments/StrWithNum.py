'''
Created on Dec 23, 2017

@author: dtitx
'''

class StrWithNum:
  '''
  A wrapper around strings that allows correct comparison of strings containing numbers, like 'x2' < 'x10'.

  Parameters
  ----------
  wrapped: str
    The underlying string that is to be compared.
  '''
  def __init__(self, wrapped: str ):
    self._wrapped = wrapped

  def __cmp__(self, other) -> int:
    '''
    Old school (Python 2) comparison operator.

      * returns <0 if self < other,
      * returns  0 if self.wrapped == other.wrapped and +1
      * returns >0 if self > other
    '''
    a = self ._wrapped; i = 0
    b = other._wrapped; j = 0
    # keep track of leading zeros
    lz_cmp = []

    while( i < len(a) and
           j < len(b) ):
      if( '0' <= a[i] <= '9' and
          '0' <= b[j] <= '9' ):
        # COUNT LEADING ZEROS
        lz = 0
        while i < len(a) and '0' == a[i]: lz += 1; i += 1
        while j < len(b) and '0' == b[j]: lz -= 1; j += 1
        lz_cmp.append(lz)
        # PARSE INTEGERS
        va = 0
        vb = 0
        while i < len(a) and '0' <= a[i] <= '9': va = va*10 + int(a[i]); i += 1
        while j < len(b) and '0' <= b[j] <= '9': vb = vb*10 + int(b[j]); j += 1
        cmp = va - vb
      else:
        cmp = ord(a[i]) - ord(b[j])
        i += 1
        j += 1
      if 0 != cmp: return cmp

    # compare number of trailing chacters to resolve conflicts
    cmp = (len(a)-i) - (len(b)-j)
    if 0 != cmp: return cmp

    # compare leading zeros to resolve conflicts
    for cmp in lz_cmp:
      if 0 != cmp: return cmp

    # returns 0 if and only if self._wrapped == other._wrapped
    return 0

  def __lt__(self, other): return self.__cmp__(other) <  0
  def __le__(self, other): return self.__cmp__(other) <= 0
  def __ge__(self, other): return self.__cmp__(other) >= 0
  def __gt__(self, other): return self.__cmp__(other) >  0
  def __eq__(self, other): return self.__cmp__(other) == 0
  def __ne__(self, other): return self.__cmp__(other) != 0



assert StrWithNum('int2')  < StrWithNum('int12')
assert StrWithNum('a1b20') > StrWithNum('a1b1')
assert StrWithNum('a02b02') > StrWithNum('a2b02')
assert StrWithNum('a1b02') < StrWithNum('a01b02')
assert StrWithNum('a1b02a') > StrWithNum('a01b02')
assert StrWithNum('a1b001') < StrWithNum('a1b2')
assert StrWithNum('a2b1') == StrWithNum('a2b1')
assert StrWithNum('a2b1x') > StrWithNum('a2b1')
assert StrWithNum('b2b1') > StrWithNum('a2b1')


