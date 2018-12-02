'''
Created on Dec 25, 2017

@author: Dirk Toewe
'''

import numpy as np
from tf2x.utils import array2str


def main():
  print( array2str(np.random.rand(50,20,30,10)) )



if __name__ == '__main__':
  main()