'''
Created on Nov 11, 2017

@author: Dirk Toewe
'''
from collections import OrderedDict
from itertools import starmap

from pkg_resources import resource_string

import numpy as np, os, tensorflow as tf
from _collections import defaultdict
import json
from base64 import b64encode
from tf2x import nd


_tf2js_template = resource_string(__package__, 'tf2js.js.template').decode('utf-8')

class _no_session:
  '''
  A dummy object used to throw an exception when a tensor value is requested without a value present.
  '''
  def run( self, *args, **kwargs ):
    raise ValueError("If the Tensorflow graph contains variables, a tf.Session must be supplied via the 'sess' kwarg.")

_no_session = _no_session()

def tensor2js( tensor: tf.Tensor, *, sess: tf.Session=_no_session, model_name: str='Model' ):
  '''
  Converts a Tensorflow Graph into the string representation of a JavaScript class definition.

  This conversion method does not support control flow/inputs.

  The generated class depends on nd.js, a tiny, rudimentary JavaScript library that implements
  basic NumPy-like array operations.

  Parameters
  ----------
  tensor: tf.Tensor
    The tensorflow computation that is to be converted to JavaScript.
  sess: tf.Session
    Used to read Variable values data from the model. Can be omitted if there
    are not Variables in the graph.
  model_name: str
    The name of the generated JavaScript class. Said class represents
    the computation in JS.

  Returns
  -------
  result: str
    A JavaScript class as string, whose instances are functions, which can be call with a feed_dict
    as input to evaluate the tensor. The model class depends on the nd.js library.
  '''

  vars, inputs, consts, ops = ( OrderedDict() for _ in range(4) )

  tensors = {}
  refs = defaultdict(lambda: 0)

  class mkVal:
    def Const(tensor):
      val = sess.run(tensor)
      shape = tensor.shape
      dtype = tensor.dtype.name
      consts[tensor.name] = (dtype,shape,val)
      return "model.consts['%s']" % tensor.name

    def Placeholder(tensor):
      shape = tensor.shape
      dtype = tensor.dtype.name
      inputs[tensor.name] = (dtype,shape)
      return "inputs['%s']" % tensor.name

    def VariableV2(tensor):
      val = sess.run(tensor)
      shape = tensor.shape
      dtype = tensor.dtype.name[:-4]
      vars[tensor.name] = (dtype,shape,val)
      return "model.vars['%s']" % tensor.name
  mkVal = mkVal.__dict__


  class mkOp:
    def Add    (op, in0,in1): return "nd.Array.from([%s,%s], '%s', (x,y) => x+y )" % (in0, in1, op.outputs[0].dtype.name)
    def Sub    (op, in0,in1): return "nd.Array.from([%s,%s], '%s', (x,y) => x-y )" % (in0, in1, op.outputs[0].dtype.name)
    def Mul    (op, in0,in1): return "nd.Array.from([%s,%s], '%s', (x,y) => x*y )" % (in0, in1, op.outputs[0].dtype.name)
    def RealDiv(op, in0,in1): return "nd.Array.from([%s,%s], '%s', (x,y) => x/y )" % (in0, in1, op.outputs[0].dtype.name)

    def Exp     (op, in0): return "%s.map( '%s', x =>     Math.exp(x)  )" % (in0, op.outputs[0].dtype.name)
    def Relu    (op, in0): return "%s.map( '%s', x =>     Math.max(0,x))" % (in0, op.outputs[0].dtype.name)
    def Rsqrt   (op, in0): return "%s.map( '%s', x => 1 / Math.sqrt(x) )" % (in0, op.outputs[0].dtype.name)
    def Shape   (_,  in0): return "nd.array(%s.shape, 'int32')"  % in0
    def Identity(_,  in0): return in0

    def Softmax(op, in0): return "_softmax(%s, '%s')" % (in0,op.outputs[0].dtype.name)

    def Sum(op, tensor, axes): return "%s.reduce(%s, '%s', (x,y) => x+y )" % (tensor,axes, op.outputs[0].dtype.name)

    def Reshape(_, tensor, shape): return '_reshape(%s,%s)' % (tensor,shape)

    def StridedSlice(op, *params):
      assert 4 == len(params)
      params = (
        *params,
        *(
          '/*%s=*/' % key + bin( op.get_attr(key) )
          for key in ['begin_mask','end_mask', 'ellipsis_mask', 'new_axis_mask', 'shrink_axis_mask']
        )
      )
      return '_stridedSlice(%s)' % ', '.join(params)

    def Pack(op, *tensors):
      axis = op.get_attr('axis')
      return "nd.stack(%s, '%s', [%s])" %(axis, op.outputs[0].dtype.name, ', '.join(tensors) )

    def Conv2D(op, images, filter):
      strides    = op.get_attr('strides')
      padding    = op.get_attr('padding'    ).decode('ascii') # <- TODO: utf-8?
      data_format= op.get_attr('data_format').decode('ascii')
      return "_conv2d(%s, %s, %s, '%s', '%s')" % (images,filter,strides,padding,data_format)
  mkOp = mkOp.__dict__


  def convert( tensor: tf.Tensor ):
    if tensor in tensors:
      return tensors[tensor]
    op = tensor.op
    assert 1 == len(op.outputs)
    assert 0 == len(op.control_inputs)
    if op.type in mkVal:
      tensors[tensor] = mkVal[op.type](tensor)
    else:
      for input in op.inputs: refs[input.name] += 1
      expr = mkOp[op.type]( op, *map(convert, op.inputs) )
      ops[tensor.name] = expr
      tensors[tensor] = "model.ops['%s'](inputs)" % tensor.name
    return tensors[tensor]

  refs[tensor.name] += 1
  result = convert(tensor)
# 
#   print('------')
#   print('ops=')
#   print( '\n'.join( starmap( '  {} \t-> {}'.format, ops.items() ) ) )
#   print('------')
#   print('vars=')
#   print( '\n'.join( starmap( '  {} \t-> {}'.format, vars.items() ) ) )
#   print('------')
#   print('consts=')
#   print( '\n'.join( starmap( '  {} \t-> {}'.format, consts.items() ) ) )
#   print('------')
#   print('inputs=')
#   print( '\n'.join( starmap( '  {} \t-> {}'.format, inputs.items() ) ) )
#   print('------')
#   print( json.dumps(refs) )
#   print('------')

  def shapeStr( shape: tf.TensorShape ):
    '''
    Returns (as string) a JavaScript representation of a np.NDArray shape. Unspecified dimensions are
    represented as undefined in JavaScript.
    '''
    if None is shape.dims:
      return 'undefined'

    shape = (
      'undefined' if None is dim.value else str(dim.value)
      for dim in shape
    )
    return '[' + ', '.join(shape) + ']'


  result = _tf2js_template.format(
    MODEL_NAME = model_name,
    RESULT = result,
    CONSTS = ',\n      '.join( "{:32s}: {}"          .format("'%s'" % name,nd.arrayB64(vals).replace('\n', '\n      ')) for name,(_,_,vals) in consts.items() ),
    VARS   = ',\n      '.join( "{:32s}: {}"          .format("'%s'" % name,nd.arrayB64(vals).replace('\n', '\n      ')) for name,(_,_,vals) in   vars.items() ),
    OPS    = ',\n      '.join( "{:32s}: inputs => {}".format("'%s'" % name,expr)                                        for name,expr       in    ops.items() ),
    CHECKS =  '\n      '.join(
      '_checkNDArray({0}, "{0}", \'{1}\', {2});'.format( "%s['%s']" % (kind,name), type, shapeStr(shape) )
      for kind,items in [
        ('model.vars',   ( (name,(typ,shape)) for name,(typ,shape,_) in vars  .items() ) ),
        ('model.consts', ( (name,(typ,shape)) for name,(typ,shape,_) in consts.items() ) ),
        ('inputs', inputs.items() )
      ]
      for name,(type,shape) in items
    ),
    REFS = json.dumps(refs)
  )
 
  return result

