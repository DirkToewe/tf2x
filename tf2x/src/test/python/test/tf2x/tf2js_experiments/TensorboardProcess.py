from multiprocessing import Process

class TensorboardProcess(Process):

  def __init__(self, *, logdir ):
    super(TensorboardProcess,self).__init__()
    self.logdir = logdir

  def run(self):
    import tensorflow.tensorboard.tensorboard
    import sys
    sys.argv = ['', '--logdir='+self.logdir]
    sys.exit(tensorflow.tensorboard.tensorboard.main())