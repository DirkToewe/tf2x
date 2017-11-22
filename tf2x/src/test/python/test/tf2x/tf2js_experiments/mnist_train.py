'''
A fairly simple convolutional neural network (CNN) trained to classify
MNIST hand-written decimal digits. The CNN is heavily inspired by Martin
Görner's talk "Tensorflow and deep learning without a PhD".

Created on Sep 11, 2017

@author: Dirk Toewe
'''

import logging, numpy as np, os, tensorflow as tf, webbrowser

from scipy.misc import imsave
from tensorflow import newaxis as tf_new
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

from test.tf2x.tf2js_experiments.ImageDistorter import ImageDistorter
from test.tf2x.tf2js_experiments.TensorboardProcess import TensorboardProcess

# FIXME:
# In practice, the algorithm perfoms terribly especially, if the numbers tested touch the image border
#
# * Introduce some baseline of noisy black images that are to be classified as 'NaN'
# * Introduce some random movement of the images
#
# The MNIST dataset it terrible for "german" digit notation.

def relu( input ):
  '''
  Applies the activation function to the given input (layer).

  Parameters
  ----------
  input: tf.Tensor[...]

  Returns
  -------
  activated: tf.Tensor[...]
  '''
  tf.summary.histogram('logits', input)
  return input*1e-3 + tf.nn.relu(input)



def sig( input ):
  tf.summary.histogram('logits', input)
  return input*1e-3 + tf.nn.sigmoid(input)



class Model:
  
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
  # 99.5% BEGIN
#     with tf.name_scope('var'):
#       var_l1_filter = tf.Variable( name='l1_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,4,24],  stddev=0.1) )
#       var_l1_bias   = tf.Variable( name='l1_bias',   dtype=tf.float32, initial_value=tf.truncated_normal([24],        stddev=0.1) )
#                                                                                                                              
#       var_l2_filter = tf.Variable( name='l2_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,24,24], stddev=0.1) )
#       var_l2_bias   = tf.Variable( name='l2_bias',   dtype=tf.float32, initial_value=tf.truncated_normal([11,11,24],  stddev=0.1) )
#                                                                                                                              
#       var_l3_filter = tf.Variable( name='l3_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,24,24], stddev=0.1) )
#       var_l3_bias   = tf.Variable( name='l3_bias',   dtype=tf.float32, initial_value=tf.truncated_normal([6,6,24],    stddev=0.1) )
#       
#       var_l4_weights= tf.Variable( name='l4_weights',dtype=tf.float32, initial_value=tf.truncated_normal([6,6,24,196],stddev=0.1) )
#       var_l4_bias   = tf.Variable( name='l4_bias',   dtype=tf.float32, initial_value=tf.truncated_normal([196],       stddev=0.1) )
#       
#       var_l5_weights= tf.Variable( name='l5_weights',dtype=tf.float32, initial_value=np.zeros([196,10]) )
#       var_l5_bias   = tf.Variable( name='l5_bias',   dtype=tf.float32, initial_value=np.zeros([10],   ) )
  # 99.5% END
    with tf.name_scope('var'):
      var_l1_filter = tf.Variable( name='l1_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,1,16], stddev=0.1) )
      var_l1_bias   = tf.Variable( name='l1_bias',   dtype=tf.float32, initial_value=tf.truncated_normal([16],       stddev=0.1) )
                                                                                                                             
      var_l2_filter = tf.Variable( name='l2_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,16,16],stddev=0.1) )
      var_l2_bias   = tf.Variable( name='l2_bias',   dtype=tf.float32, initial_value=tf.truncated_normal([16],       stddev=0.1) )
                                                                                                                             
      var_l3_filter = tf.Variable( name='l3_filter', dtype=tf.float32, initial_value=tf.truncated_normal([7,7,16,8], stddev=0.1) )
      var_l3_bias   = tf.Variable( name='l3_bias',   dtype=tf.float32, initial_value=tf.truncated_normal([6,6,8],    stddev=0.1) )
      
      var_l4_weights= tf.Variable( name='l4_weights',dtype=tf.float32, initial_value=tf.truncated_normal([6,6,8,160],stddev=0.1) )
      var_l4_bias   = tf.Variable( name='l4_bias',   dtype=tf.float32, initial_value=tf.truncated_normal([160],      stddev=0.1) )
      
      var_l5_weights= tf.Variable( name='l5_weights',dtype=tf.float32, initial_value=np.zeros([160,10]) )
  
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
      # MIRROR THE IMAGES ALONG ALL AXES TO MAKE SYMMETRY INFORMATION AVAILABLE FOR THE CONVOLUTIONAL KERNELS
      in_images_stack = tf.reshape(in_images, [-1,h,w,1])
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
      with tf.name_scope('l1'): op_l1,upd1=bnorm(tf.nn.conv2d( in_images_stack,  var_l1_filter, [1,1,1,1],'VALID'),var_l1_bias,conv=True ); op_l1=relu(op_l1); print('layer1.shape: %s' % op_l1.shape)
      with tf.name_scope('l2'): op_l2,upd2=bnorm(tf.nn.conv2d( op_l1,            var_l2_filter, [1,2,2,1], 'SAME'),var_l2_bias,conv=True ); op_l2=relu(op_l2); print('layer2.shape: %s' % op_l2.shape)
      with tf.name_scope('l3'): op_l3,upd3=bnorm(tf.nn.conv2d( op_l2,            var_l3_filter, [1,2,2,1], 'SAME'),var_l3_bias,conv=True ); op_l3=relu(op_l3); print('layer3.shape: %s' % op_l3.shape)
      with tf.name_scope('l4'): op_l4,upd4=bnorm(tf.reduce_sum(op_l3[...,tf_new]*var_l4_weights, axis=[-2,-3,-4] ),var_l4_bias,conv=False); op_l4=relu(op_l4); print('layer4.shape: %s' % op_l4.shape)
      with tf.name_scope('l5'): op_l5,upd5=bnorm(tf.reduce_sum(op_l4[...,tf_new]*var_l5_weights, axis=[-2]       ),            conv=False);                    print('layer5.shape: %s' % op_l5.shape)
  
      print()
      self.batchnorm_update = tf.group(upd1,upd2,upd3,upd4,upd5)
  
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
      train_rate         = tf.train.exponential_decay(1e-2, train_step, 100, 0.95)
      self.train_minimize= tf.train.AdamOptimizer(train_rate).minimize( out_loss, train_step, var_list=var_list )
    tf.summary.scalar('train_rate', train_rate)



def main():
  ##
  ## CREATE MODEL
  ##
  w,h = 28,28
  model = Model(w,h)
#   out_images_distorted = tf.identity(model.in_images)
  out_images_distorted = tf.map_fn(ImageDistorter(), model.in_images)

  ##
  ## READ DATA
  ##
  mnist = mnist_data.read_data_sets( os.path.expanduser('~/Pictures/MNIST/data'), one_hot=True)

  in_images_train     = mnist.train     .images.reshape(-1,h,w,1)
#   in_images_validation= mnist.validation.images.reshape(-1,h,w,1)
  in_images_test      = mnist.test      .images.reshape(-1,h,w,1) # <- used to estimate the model error during training
    
  in_labels_train     = mnist.train     .labels
#   in_labels_validation= mnist.validation.labels
  in_labels_test      = mnist.test      .labels # <- used to estimate the model error during training

  ##
  ## SUMMARY DATA: used to inspect and debug model and training
  ##
  summary_dir = os.path.expanduser('~/Pictures/MNIST/summary')
  summary_out = tf.summary.merge_all()
  summary_log = tf.summary.FileWriter(summary_dir)
  summary_embed_size = 32

  summary_sprites_path = os.path.expanduser('~/Pictures/MNIST/summary/sprites.png')
  summary_sprites = np.row_stack(
    np.column_stack( col for col in row )
    for row in in_images_test[:summary_embed_size**2].reshape(summary_embed_size,summary_embed_size,h,w)
  )
  summary_sprites = 1 - summary_sprites
#   summary_sprites = np.repeat(summary_sprites[:,:,np.newaxis],4, axis=2)
#   summary_sprites[:,:,:3] = 0
  imsave(summary_sprites_path, summary_sprites)

  summary_embed_size **= 2
  summary_labels_path = os.path.expanduser('~/Pictures/MNIST/summary/sprites.tsv')
  with open(summary_labels_path, 'w') as out:
    out.write( '\n'.join(
      '%d' % label
      for label in np.argmax( in_labels_test[:summary_embed_size], axis=1 )
    ) )
  summary_embed_shape = [summary_embed_size,10]
  summary_embed = tf.Variable(name='embedding', initial_value=np.zeros(summary_embed_shape), dtype=tf.float32)
  summary_embed_update = summary_embed.assign(model.out_prediction)
  summary_embed_proj = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  summary_embed_cfg = summary_embed_proj.embeddings.add()
  summary_embed_cfg.tensor_name = summary_embed.name
#   summary_embed_cfg.tensor_shape.extend(summary_embed_shape)
  summary_embed_cfg.metadata_path = 'sprites.tsv'
  summary_embed_cfg.sprite.image_path = 'sprites.png'
  summary_embed_cfg.sprite.single_image_dim.extend([h,w])
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_log, summary_embed_proj)

  ##
  ## CREATE SAVER
  ##
  saver = tf.train.Saver( keep_checkpoint_every_n_hours=1 )

  TensorboardProcess(logdir = '~/Pictures/MNIST/summary/').start()
  webbrowser.open('http://localhost:6006')

  ##
  ## TRAIN MODEL
  ##
  with tf.Session() as sess:

    sess.run( tf.global_variables_initializer() )
    summary_log.add_graph(sess.graph)

    model_dir = os.path.expanduser('~/Pictures/MNIST/summary/model.ckpt')

    try:
      load_path = model_dir + "-2240"
      saver.restore(sess, load_path)
    except tf.errors.NotFoundError as nfe:
      logging.warn(nfe)

    print()
    print('training...')

    def train( in_image_train, in_labels_train ):

      for _ in range(1):
        train_feed = {
          'in/images:0': sess.run(out_images_distorted, feed_dict = { 'in/images:0': in_image_train }),
          'in/labels:0': in_labels_train,
          'in/train_mode:0': True
        }
  
        # train a few steps with the batch
        for _ in range(1):
          sess.run( model.train_minimize,   feed_dict=train_feed )
          sess.run( model.batchnorm_update, feed_dict=train_feed )

    test_feed = {
      'in/images:0': in_images_test,
      'in/labels:0': in_labels_test,
      'in/train_mode:0': False
    }

    def summarize(step):
      loss, n_errors, accuracy, summ = sess.run(
        [model.out_loss, model.out_n_errors, model.out_accuracy, summary_out],
        feed_dict=test_feed
      )
      print( '  xentropy: %f' % loss )
      print( '  #errors: %d' % n_errors )
      print( '  accuracy: %f%%' % (accuracy*100) )
      summary_log.add_summary(summ,step)

    order = np.arange( len(in_images_train) )

    for _ in range(1000*1000):

      np.random.shuffle(order)

      batch_size = 2500

      for batch_indices in order.reshape(-1,batch_size):

        images = in_images_train[batch_indices]
        labels = in_labels_train[batch_indices]

        step = sess.run(model.train_step) + 1
        print('ITERATION%5d' % step)
        train(images,labels)

        if step % 10 == 0:
          print('--------------')
          summarize(step)
          print()

        if step % 20 == 0:
          print('  saving...', end='')
          print(' done!')
          sess.run(
            summary_embed_update,
            feed_dict = {
              'in/images:0': in_images_test[:summary_embed_size],
              'in/train_mode:0': False 
            }
          )
          print()
          saver.save(sess, model_dir, global_step=step)



if __name__ == '__main__':
  main()


