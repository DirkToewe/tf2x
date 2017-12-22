'''
Created on Oct 31, 2017

@author: Dirk Toewe
'''
from typing import Union, Set

from graphviz import Digraph

import numpy as np, tensorflow as tf
from tf2x.inspect import ops
import logging


def _ops2dot( ops: Set[tf.Operation], *, sess: tf.Session=None ):

  dot = Digraph()

  name2subgraph = {}
  name2node = {}

  def fetch_subgraph( name: str ):
    if name in name2subgraph:
      return name2subgraph[name]

    if 0 == len(name):
      return dot

    namespace = name.rsplit('/',1)
    if 1 == len(namespace):
      result = dot.subgraph(name='cluster_'+name, graph_attr={'label':name})
      name2subgraph[name] = result
      return result
    namespace,_ = namespace

    outer = fetch_subgraph(namespace)
    with fetch_subgraph(namespace) as graph:
      inner = graph.subgraph(name='cluster_'+name, graph_attr={'label':name})
      class result:
        @staticmethod
        def __enter__():
          outer.__enter__()
          return inner.__enter__()
        @staticmethod
        def __exit__ (type_, value, taceback):
          inner.__exit__(type_, value, taceback)
          outer.__exit__(type_, value, taceback)
      result = result()
      name2subgraph[name] = result
      return result

  def fetch_node( name: str, **attrs ):
    if name in name2node:
      assert 0 == len(attrs)
      return name2node[name]

    namespace = name.rsplit('/',1)
    if 1 == len(namespace):
      dot.node(name, **attrs)
      name2node[name] = name
      return name
    namespace,_ = namespace

    with fetch_subgraph(namespace) as subgraph:
      subgraph.node(name, **attrs)
      name2node[name] = name
      return name

  ##
  ## ADD OPERATION NODES
  ##
  for op in ops:
    n_in_out = max(
      len(op.inputs),
      len(op.outputs)
    )

    def label():
      #
      # INPORTS
      #
      n_in = len(op.inputs)
      ports = ''.join( '<td border="1" port="in{I}">in{I}</td>'.format(I=i) for i in range(n_in) )
      left = n_in_out - n_in
      left,right = 2 + left//2, 2 - left//2 + left
      yield '<td colspan="{LEFT}"></td>{PORTS}<td colspan="{RIGHT}"></td>'.format(LEFT=left, PORTS=ports, RIGHT=right)

      #
      # CONTROL PORTS
      #
      n_cin = len(op.control_inputs)

      label = '&lt;{TYPE}&gt;<br/><b>{NAME}</b>'.format(
        TYPE = op.type,
        NAME = op.name,
      )
      if None is not sess:
#       if op.type == 'Const' and None is not sess:
        threshold = np.get_printoptions()['threshold']
        try:
          np.set_printoptions(threshold=16)
          label += '<br/>={}'.format( np.array2string( sess.run(op.outputs[0]), separator=', ' ).replace('\n','<br/>') )
        except:
          logging.warning('Could not read %s "%s".' % (op.type,op.name) )
        finally:
          np.set_printoptions(threshold=threshold)

      yield '<td></td><td rowspan="{ROWSPAN}" cellpadding="8" colspan="{COLSPAN}">{LABEL}</td><td></td>'.format(
        LABEL = label,
        ROWSPAN = 2+max(1,n_cin),
        COLSPAN = 2+n_in_out
      )

      if 0 == n_cin:
        yield '<td></td><td border="1" port="cout">cout</td>'
      else:
        yield '<td border="1" port="cin0">cin0</td><td rowspan="{ROWSPAN}" border="1" port="cout">cout</td>'.format(ROWSPAN=n_cin)
        for i in range(1,n_cin):
          yield '<td port="cin{I}">{I}</td>'.format(I=i)

      yield '<td></td><td></td>'

      #
      # OUTPORTS
      #
      n_out = len(op.outputs)
      left = n_in_out - n_out
      left,right = 2 + left//2, 2 - left//2 + left
      yield '<td colspan="{LEFT}"></td>{PORTS}<td colspan="{RIGHT}"></td>'.format(
        LEFT=left,
        PORTS=''.join( '<td border="1" port="out{I}">out{I}</td>'.format(I=i) for i in range(n_out) ),
        RIGHT=right,
        COLSPAN=n_out
      )
      

    label = '</tr><tr>'.join( label() )
    label = '<<font face="Ubuntu"><table border="2" cellborder="0" cellspacing="0" bgcolor="#FFCC00"><tr>%s</tr></table></font>>' % label
#     label = '<<table border="2" cellborder="0" cellspacing="0" bgcolor="#FFCC00"><tr>%s</tr></table>>' % label

    fetch_node(op.name, label=label, shape='none')#label='<%s>\n%s' % (op.type,op.name) )

  for target in ops:
    targetNode = fetch_node(target.name)
  
    ##
    ## ADD DATA FLOW EDGES
    ##
    for i,source in enumerate(target.inputs):
      sourceNode = fetch_node(source.op.name)

      srcShape = '[...]'
      if None is not source.shape.dims: # <- FIXME
        srcShape = ','.join( ':' if None is d else str(d) for d in source.shape )
      srcShape = '[%s]' % srcShape
      label = str(source.dtype.name) + srcShape

      dot.edge( sourceNode+':out%d' % source.value_index, targetNode+':in%d' % i, label=label )

    ##
    ## ADD CONTROL FLOW EDGES
    ##
    for i,source in enumerate(target.control_inputs):
      sourceNode = fetch_node(source.name)
      dot.edge(sourceNode+':cout', targetNode+':cin%d' % i, style='dashed', label='\<ctrl\>')

  return dot


def tf2dot( parent: Union[tf.Tensor,tf.Graph,tf.Operation], *, sess: tf.Session=None ):
  '''
  Returns a DOT graph representation of the Tensorflow computation associated with the given parent object.

    * If parent is a tensor, a graph representation is created for all operations the tensor depends on.
    * If parent is a graph, a graph representation is created for its operations.
    * If parent is an operation, a graph representation is created for the op and all its dependencies.

  Parameters
  ----------
  parent: tf.Tensor or tf.Grpah or tf.Operation

  Returns
  -------
  graphviz.Digraph
    An DOT graph representation of the computation (sub)graph.
  '''
  return _ops2dot( ops(parent), sess=sess )