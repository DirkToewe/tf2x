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
from test.tf2x.tf2js_experiments.MNIST_Model import MNIST_Model

# FIXME:
# In practice, the algorithm perfoms terribly especially, if the numbers tested touch the image border
#
# * Introduce some baseline of noisy black images that are to be classified as 'NaN'
# * Introduce some random movement of the images
#
# The MNIST dataset it terrible for "german" digit notation.
def main():
  ##
  ## CREATE MODEL
  ##
  w,h = 28,28
  model = MNIST_Model(w,h)

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
      load_path = model_dir + "-3800"
      saver.restore(sess, load_path)
    except tf.errors.NotFoundError as nfe:
      logging.warn(nfe)

    print()
    print('training...')

    def train( in_images_train, in_labels_train ):

      for _ in range(1):
        train_feed = {
          model.in_images: in_images_train,
          model.in_labels: in_labels_train,
          model.in_train_mode: True
        }
        sess.run( model.train, feed_dict=train_feed )

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

        step = sess.run(model.train_step)

        if step % 5 == 0:
          print('ITERATION%5d' % step)
        train(images,labels)

        if 0 < step and step % 10 == 0:
          print('--------------')
          summarize(step)
          print()

        if 0 < step and step % 20 == 0:
          print('  saving embedding...', end='')
          sess.run(
            summary_embed_update,
            feed_dict = {
              'in/images:0': in_images_test[:summary_embed_size],
              'in/train_mode:0': False 
            }
          )
          print(' done!')
          print()

        if 0 < step and step % 100 == 0:
          print('  saving graph...', end='')
          saver.save(sess, model_dir, global_step=step)
          print(' done!')
          print()



if __name__ == '__main__':
  main()


