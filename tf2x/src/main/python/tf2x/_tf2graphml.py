'''
Created on Oct 28, 2017

@author: Dirk Toewe
'''

from typing import Union, Set

import tensorflow as tf
from tf2x.utils import ops
import xml.etree.ElementTree as xml


# TODO: make the yEd stuff optional
def _ops2graphml( ops: Set[tf.Operation] ):
  NAMESPACES = dict(
    gml = 'http://graphml.graphdrawing.org/xmlns',
    yed = 'http://www.yworks.com/xml/graphml'
  )

  GML = '{http://graphml.graphdrawing.org/xmlns}%s'
  XSI = '{http://www.w3.org/2001/XMLSchema-instance}%s'
  YED = '{http://www.yworks.com/xml/graphml}%s'
  
  xml.register_namespace('gml', 'http://graphml.graphdrawing.org/xmlns')
  xml.register_namespace('yed', 'http://www.yworks.com/xml/graphml')
  doc = xml.Element(
    GML % 'graphml',
    **{XSI % 'schemaLocation': 'http://graphml.graphdrawing.org/xmlns    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd    http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd'}
  )
 
  xml.SubElement(doc, GML % 'key', attrib={'for': 'node', 'id': 'node$description', 'attr.type': 'string', 'attr.name': 'description' } )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'node', 'id': 'node$name',        'attr.type': 'string', 'attr.name': 'name'        } )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'node', 'id': 'node$type',        'attr.type': 'string', 'attr.name': 'type'        } )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'edge', 'id': 'edge$type',        'attr.type': 'string', 'attr.name': 'type'        } )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'edge', 'id': 'edge$dtype',       'attr.type': 'string', 'attr.name': 'dtype'       } )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'edge', 'id': 'edge$shape',       'attr.type': 'string', 'attr.name': 'shape'       } )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'edge', 'id': 'edge$index_source','attr.type': 'int',    'attr.name': 'index_source'} )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'edge', 'id': 'edge$index_target','attr.type': 'int',    'attr.name': 'index_target'} )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'node', 'id': 'node$gfx',       'yfiles.type': 'nodegraphics'                       } )
  xml.SubElement(doc, GML % 'key', attrib={'for': 'edge', 'id': 'edge$gfx',       'yfiles.type': 'edgegraphics'                       } )

  graph = xml.SubElement(doc, GML % 'graph', edgedefault="directed")

  name2elem = {}

  def fetch_elem( name: str ):
    '''
    Returns the node-(xml)-element of the group containing the given operator (name).
    '''
    if name not in name2elem:
      split = name.rsplit('/',1)

      if 1 == len(split): parent = doc
      else:               parent = fetch_elem(split[0])

      parent_graph = parent.find('gml:graph', NAMESPACES)
      if None is parent_graph:
        parent_graph = xml.SubElement(parent, GML % 'graph')

      elem = xml.SubElement(parent_graph, GML % 'node', id=name)
      name2elem[name] = elem

      datName = xml.SubElement(elem, GML % 'data', key='node$name')
      datType = xml.SubElement(elem, GML % 'data', key='node$type')
      datName.text = name
      datType.text = '<group>'

    return name2elem[name]

  ##
  ## ADD OPERATION NODES
  ##
  for op in ops:
    elem = fetch_elem(op.name)

    datName = elem.find('./gml:data[@key="node$type"]', NAMESPACES)
    datName.text = op.type

    datGFX     = xml.SubElement(elem,       GML % 'data', key='node$gfx')
    datGFXShape= xml.SubElement(datGFX,     YED % 'ShapeNode')
    datGFXGeom = xml.SubElement(datGFXShape,YED % 'Geometry', width='128', height='64')
    datGFXText = xml.SubElement(datGFXShape,YED % 'NodeLabel')
    datGFXText.text =  '<%s>\n"%s"' % (op.type,op.name)

    for i,_ in enumerate(op.        inputs): xml.SubElement(elem, GML % 'port', name=  'in:%d' % i)
    for i,_ in enumerate(op.control_inputs): xml.SubElement(elem, GML % 'port', name='ctrl:%d' % i)
    for i,_ in enumerate(op.       outputs): xml.SubElement(elem, GML % 'port', name= 'out:%d' % i)

  ##
  ## ADD EDGES
  ##
  for op in ops:
    elem = fetch_elem(op.name)

    ##
    ## ADD DATA FLOW EDGES
    ##
    for i,src in enumerate(op.inputs):
      edge = xml.SubElement(
        graph, GML % 'edge',
        source=src.op.name, sourceport="out:%d" % src.value_index,
        target=op.name,     targetport="in:%d" % i
      )

      srcShape = '[...]'
      if None is not src.shape: # <- FIXME
        srcShape = ','.join( ':' if None is d else str(d) for d in src.shape )
      srcShape = '[%s]' % srcShape

      datType = xml.SubElement(edge, GML % 'data', key='edge$type' )
      datDType= xml.SubElement(edge, GML % 'data', key='edge$dtype')
      datShape= xml.SubElement(edge, GML % 'data', key='edge$shape')
      datIS   = xml.SubElement(edge, GML % 'data', key='edge$index_source')
      datIT   = xml.SubElement(edge, GML % 'data', key='edge$index_target')
      datType .text = str(src.dtype.name) + srcShape
      datDType.text = str(src.dtype.name)
      datShape.text = srcShape
      datIS.text = str(src.value_index)
      datIT.text = str(i)

      datGFX     = xml.SubElement(edge,      GML % 'data', key='edge$gfx')
      datGFXPoly = xml.SubElement(datGFX,    YED % 'PolyLineEdge')
      datGFXArrow= xml.SubElement(datGFXPoly,YED % 'Arrows', target='standard')
      datGFXLabel= xml.SubElement(datGFXPoly,YED % 'EdgeLabel')
      datGFXLabel.text = str(src.dtype.name) + srcShape

    ##
    ## ADD CONTROL FLOW EDGES
    ##
    for i,src in enumerate(op.control_inputs):
      edge = xml.SubElement(
        graph, GML % 'edge',
        source=src.name,
        target=op.name,
        targetport="ctrl:%d" % i
      )

      datType = xml.SubElement(edge, GML % 'data', key='edge$type')
      datType.text = '<control>'

      datGFX     = xml.SubElement(edge,      GML % 'data', key='edge$gfx')
      datGFXPoly = xml.SubElement(datGFX,    YED % 'PolyLineEdge')
      datGFXStyle= xml.SubElement(datGFXPoly,YED % 'LineStyle', type='dashed')
      datGFXArrow= xml.SubElement(datGFXPoly,YED % 'Arrows', target='diamond')
      datGFXLabel= xml.SubElement(datGFXPoly,YED % 'EdgeLabel')
      datGFXLabel.text = '<ctrl>'

  return xml.ElementTree(doc)
  

def tf2graphml( parent: Union[tf.Tensor,tf.Graph,tf.Operation] ):
  '''
  Returns a GraphML representation of the Tensorflow computation associated with the given parent object.

    * If parent is a tensor, a graph representation is created for all operations the tensor depends on.
    * If parent is a graph, a graph representation is created for its operations.
    * If parent is an operation, a graph representation is created for the op and all its dependencies.

  The resulting XML document can be opened and layed out in yEd directly as it contains
  the respective yEd metadata.

  Parameters
  ----------
  parent: tf.Tensor or tf.Graph or tf.Operation

  Returns
  -------
  graphml: xml.ElementTree
    An XML/GraphML representation of the computation (sub)graph.
  '''
  return _ops2graphml( ops(parent) )

