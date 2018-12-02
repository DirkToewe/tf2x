TF2X is a self-educational project, exploring the computation graph
and the the underlying computational model of Tensorflow. As the name
suggests, the idea was to traverse the computation graph and convert it
to other representations.

TF2JS
-----
One rather straight forward idea was to convert to the computation
graph into executable code, e.g. JavaScript (in my defense I initiated
this project shortly *before* Tensorflow.JS was announced). TF2X offers
the `tf2js` method, that converts a Tensorflow graph to JavaScript code.
As of yet only a small subset of operations is supported but that subset
can be extended fairly easily.

The `tf2js` method takes a TF graph and converts it to a string containing
a JS class definition. Said class can be instantiated and then executed to
compute the individual graph nodes. The following example demonstates the
conversion from TF graph to JS and how to execute the result in NodeJS:

```python
import numpy as np, os, subprocess, tensorflow as tf, tf2x
from pkg_resources import resource_string
from tempfile import mkdtemp
from tf2x import nd, tensor2js

ndjs_code = resource_string(tf2x.__name__, 'nd.js').decode('utf-8')

js_code_template = '''
{ND}
{Model}
const model = new MyModel()
const a_data = {a_data}
let output = model({{ 'a:0': a_data }})
console.log('\\nOutput JS:')
console.log( output.toString() )
'''

tf_a = tf.placeholder(name='a', shape=[3], dtype=tf.float32)
tf_b = tf.constant(10, name='b', dtype=tf.float32)
tf_c = tf_a + tf_b

with tf.Session() as sess:
  tf_out = sess.run( tf_c, feed_dict={ tf_a: [1,2,3] } )
  print('Output TF:')
  print( np.array2string( tf_out, separator=', ', max_line_width=256 ) )

  js_code = js_code_template.format(
    ND = ndjs_code,
    Model = tensor2js(tf_c, sess=sess, model_name='MyModel'),
    a_data = nd.arrayB64( np.array([1,2,3], dtype=np.float32) )
  )

js_dir = mkdtemp()
js_file = os.path.join(js_dir, 'main.js')

with open(js_file, 'w') as fout:
  fout.write(js_code)

proc = subprocess.Popen(['node', 'main.js'], stderr=subprocess.STDOUT, cwd=js_dir)
proc.wait()
```

The TF2X also contains a small example of an 99.3% MNIST classifier that
is trained in Python and then converted to JS code. The result can be
found here and the code here. 

TF2Dot & TF2GraphML
-------------------
One of the main purposes of this project was to improve the understanding
of Tensorflow's computational model. Good visualization of the graph is key
to understanding. While the Tensorboard graph visualization gives an excellent
overview over a large graph, it hides away quite a few details about the
wiring of the graph on a low level. For those purposes, TF2X offers conversion
method for the computation graph to both the Dot and GraphML format.

Other Ideas
-----------
More ideas for the conversion of the TF graph were played around with but
not implemented:

  * TF Graph -> Spreadsheet
  * TF Graph -> LaTex Math

Exploring the Computation Graph
-------------------------------

