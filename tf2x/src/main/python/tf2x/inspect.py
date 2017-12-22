'''
A collection of utility methods for the inspection of Tensorflow computation graphs.

Created on Nov 1, 2017

@author: Dirk Toewe
'''

from typing import Union, Set

import tensorflow as tf


def tensors( tensor: tf.Tensor ) -> Set[tf.Tensor]:
  '''
  Returns all tensorflow tensors the specified tensor depends on, including
  the tensor itself.
  '''
  tensors = set()
  def fetch_tensors( tensor: tf.Tensor ):
    if tensor not in tensors:
      tensors.add(tensor)
      for t in tensor.op.inputs:
        fetch_tensors(t)
  fetch_tensors(tensor)
  return tensors



def trainable_vars( loss: tf.Tensor ) -> Set[tf.Variable]:
  '''
  Returns alls the trainable variables a loss function depends on.

  Parameters
  ----------
  loss: tf.Tensor
    The loss function for which the trainable variables are to be returned.

  Returns
  -------
  train_vars: {tf.Variable}
    A set of all trainable variables that loss depends on.
  '''
  op2var = {
    var.op: var
    for var in tf.trainable_variables()
  }
  return {
    op2var[tensor.op]
    for tensor in tensors(loss)
    if tensor.op in op2var
  }



def ops( group: Union[tf.Tensor,tf.Graph,tf.Operation] ) -> Set[tf.Operation]:
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


