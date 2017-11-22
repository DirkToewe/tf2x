'''
Created on Nov 1, 2017

@author: Dirk Toewe
'''

import tensorflow as tf
from typing import Union

def ops( group: Union[tf.Tensor,tf.Graph,tf.Operation] ):
  '''
  Returns all tensorflow operations associated with the given parent object.

    * If parent is a tensor, all operations the tensor depends on are returned.
    * If parent is a graph, its operations are returned.
    * If parent is an operation, the op and all its dependencies are returned.

  The resulting XML document can be opened and layed out in yEd directly as it contains
  the respective yEd metadata.

  Parameters
  ----------
  parent: tf.Tensor or tf.Graph or tf.Operation

  Returns
  -------
  graphml: set[tf.Operation]
    A set of all operations associated with parent.
  '''

  if isinstance(group, tf.Graph):
    return set(group.get_operations())

  op = group if isinstance(group, tf.Operation) else group.op

  ops = set()
  def fetch_ops( op: tf.Operation ):
    '''
    Adds a given operation and all operations it depends on to the ops set.
    '''
    if op not in ops:
      ops.add(op)
      for output in op.inputs:
        fetch_ops(output.op)
      for op in op.control_inputs:
        fetch_ops(op)

  fetch_ops(op)
  return ops