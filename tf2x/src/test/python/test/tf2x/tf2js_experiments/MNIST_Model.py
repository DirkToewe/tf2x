'''
'''

import numpy as np, tensorflow as tf
from tensorflow import newaxis as tf_new
from test.tf2x.tf2js_experiments.ImageDistorter import ImageDistorter


def _relu( logits ):
  '''
  Applies the activation function to the given input (layer).

  Parameters
  ----------
  logits: tf.Tensor[...]

  Returns
  -------
  activated: tf.Tensor[...]
  '''
  tf.summary.histogram('logits', logits)
  return logits*1e-3 + tf.nn.relu(logits)



def _sig( logits ):
  tf.summary.histogram('logits', logits)
  return logits*1e-3 + tf.nn.sigmoid(logits)



class MNIST_Model:
  
  def __init__(self, w,h, *, deploy=False):
  
    # optimization iteration index
    with tf.name_scope('train') as train_scope:
      train_step = tf.Variable(0, name='opt_step', dtype=tf.int32)
    self.train_step = train_step
  
    ##
    ## PLACEHOLDERS
    ##
    with tf.name_scope('in'):
      in_images    = tf.placeholder( name='images',    dtype=tf.float32, shape=[None,h,w,1])
      in_labels    = tf.placeholder( name='labels',    dtype=tf.float32, shape=[None,10])
      in_train_mode= tf.placeholder( name='train_mode',dtype=tf.bool,    shape=[])

    self.in_images    = in_images
    self.in_labels    = in_labels
    self.in_train_mode= in_train_mode

    if not deploy:
      in_images = tf.cond(
        in_train_mode,
        lambda: tf.map_fn(ImageDistorter(), in_images),
        lambda: in_images
      )
  
    def bnorm( wsum, bias=0, scale=1, conv=False ):
      '''
      Applies batch normalization to a neural network layer.
  
      Parameters
      ----------
      wsum: tf.Tensor[...]
        The weighted sum a.k.a. logits.
      bias: tf.Variable[...]
        The bias to be used.
      conv: bool
        Set to true for a convolutional layer, false for a fully connected layer.
        
      Returns
      -------
      out: tf.Tensor[...]
        The batch normalized and biased layer (not yet activated).
      update: () => ()
        A tensorflow computation to be executed during each training step in order
        to update the moving average used for batch normalization.
      '''
      exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, train_step) # <- adding the iteration prevents from averaging across non-existing iterations
      mean, var = tf.nn.moments(wsum, [0,1,2] if conv else [0])
    
      update = exp_moving_avg.apply([mean,var])
    
      def meanVar():
        return (
          exp_moving_avg.average(mean),
          exp_moving_avg.average(var )
        )
    
      avg_mean, avg_var = meanVar() if deploy else tf.cond(in_train_mode, lambda: (mean,var), meanVar)
      out = tf.nn.batch_normalization( wsum, avg_mean, avg_var, bias, scale, variance_epsilon=1e-8) # <- scale=1, because ReLu is linear anywas scaling won't matter
      return out, update
  
    ##
    ## VARIABLES, ORGANIZED IN LAYERS (l1, l2, l3, ...)
    ##
    with tf.name_scope('var'):
      var_l1_filter = tf.Variable( name='l1_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,1,16], stddev=0.1) )
      var_l1_bias   = tf.Variable( name='l1_bias',   dtype=tf.float32, initial_value=tf.truncated_normal(      [16], stddev=0.1) )
                                                                                                                             
      var_l2_filter = tf.Variable( name='l2_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,16,16],stddev=0.1) )
      var_l2_bias   = tf.Variable( name='l2_bias',   dtype=tf.float32, initial_value=tf.truncated_normal(       [16],stddev=0.1) )
                                                                                                                             
      var_l3_filter = tf.Variable( name='l3_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,16,8], stddev=0.1) )
      var_l3_bias   = tf.Variable( name='l3_bias',   dtype=tf.float32, initial_value=tf.truncated_normal(   [6,6,8], stddev=0.1) )
      
      var_l4_weights= tf.Variable( name='l4_weights',dtype=tf.float32, initial_value=tf.truncated_normal([6,6,8,160],stddev=0.1) )
      var_l4_bias   = tf.Variable( name='l4_bias',   dtype=tf.float32, initial_value=tf.truncated_normal(      [160],stddev=0.1) )
      
      var_l5_weights= tf.Variable( name='l5_weights',dtype=tf.float32, initial_value=np.zeros([160,10]) )
      var_l5_bias   = tf.Variable( name='l5_bias',   dtype=tf.float32, initial_value=np.zeros(    [10]) )
  
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='var/')
  
    for var in var_list:
      tf.summary.histogram(var.name, var)
  
    nVars = np.sum(
      np.prod( np.array(var.shape, dtype=np.int64) )
      for var in var_list
    )
    print()
    print('#vars = %s' % nVars)
    print()

    with tf.name_scope('op'):
      in_images_stack = tf.reshape(in_images, [-1,h,w,1])
#       # MIRROR THE IMAGES ALONG ALL AXES TO MAKE SYMMETRY INFORMATION AVAILABLE FOR THE CONVOLUTIONAL KERNELS
#       in_images_stack = tf.reshape(in_images, [-1,h,w])
#       in_images_stack = tf.stack([
#         in_images_stack,
#         in_images_stack[...,::-1, :  ],
#         in_images_stack[..., :,  ::-1],
#         in_images_stack[...,::-1,::-1]
#       ], axis=-1)
      ##
      ## OPERATIONS
      ##
      with tf.name_scope('l1'): op_l1,upd1=bnorm(tf.nn.conv2d( in_images_stack,  var_l1_filter, [1,1,1,1],'VALID'),var_l1_bias,conv=True ); op_l1=_relu(op_l1); print('layer1.shape: %s' % op_l1.shape)
      with tf.name_scope('l2'): op_l2,upd2=bnorm(tf.nn.conv2d( op_l1,            var_l2_filter, [1,2,2,1], 'SAME'),var_l2_bias,conv=True ); op_l2=_relu(op_l2); print('layer2.shape: %s' % op_l2.shape)
      with tf.name_scope('l3'): op_l3,upd3=bnorm(tf.nn.conv2d( op_l2,            var_l3_filter, [1,2,2,1], 'SAME'),var_l3_bias,conv=True ); op_l3=_relu(op_l3); print('layer3.shape: %s' % op_l3.shape)
      with tf.name_scope('l4'): op_l4,upd4=bnorm(tf.reduce_sum(op_l3[...,tf_new]*var_l4_weights, axis=[-2,-3,-4] ),var_l4_bias,conv=False); op_l4=_relu(op_l4); print('layer4.shape: %s' % op_l4.shape)
      with tf.name_scope('l5'): op_l5,upd5=bnorm(tf.reduce_sum(op_l4[...,tf_new]*var_l5_weights, axis=[-2]       ),var_l5_bias,conv=False);                     print('layer5.shape: %s' % op_l5.shape)
  
      print()
  
  #     tf.summary.image('in/images', in_images)
  #     for layer in [op_l1,op_l2,op_l3]:
  #       for i in range( layer.shape[-1] ):
  #         tf.summary.image('op/l1_%d' % i, layer[...,i,tf.newaxis] )
  
    ##
    ## OUTPUTS
    ##
    ## The prediction of the neural network for the individual images
    batch_size = tf.shape(in_labels)[0]
  
    with tf.name_scope('out'):
      out_prediction= tf.nn.softmax(op_l5, name='prediction')
      out_xentropy  = tf.nn.softmax_cross_entropy_with_logits( logits=op_l5, labels=in_labels, name='xentropy' )
      out_loss      = tf.reduce_mean(out_xentropy, name='loss')
      out_n_errors  = tf.reduce_sum(
        tf.cast(
          tf.not_equal(
            tf.argmax(in_labels,     1),
            tf.argmax(out_prediction,1)
          ),
          tf.int32
        )
      )
      out_accuracy  = tf.cast(batch_size - out_n_errors, tf.float64) / tf.cast(batch_size, tf.float64)

    self.out_prediction= out_prediction
    self.out_xentropy  = out_xentropy
    self.out_loss      = out_loss
    self.out_n_errors  = out_n_errors
    self.out_accuracy  = out_accuracy
  
    tf.summary.histogram('xentropy', out_xentropy)
    tf.summary.scalar('loss', out_loss)
    tf.summary.scalar('n_errors', out_n_errors)
    tf.summary.scalar('accuracy', out_accuracy)

    with tf.name_scope(train_scope):
      train_rate = tf.train.exponential_decay(1e-2, train_step, 100, 0.95)
      train = tf.train.AdamOptimizer(train_rate).minimize( out_loss, train_step, var_list=var_list )
      with tf.control_dependencies([train]):
        train = tf.group(upd1,upd2,upd3,upd4,upd5)
   
    self.train = train

    tf.summary.scalar('train_rate', train_rate)


