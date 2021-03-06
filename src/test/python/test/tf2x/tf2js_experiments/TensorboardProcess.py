from multiprocessing import Process

class TensorboardProcess(Process):

  def __init__(self, *, logdir ):
    super(TensorboardProcess,self).__init__()
    self.logdir = logdir

  def run(self):
    from tensorboard.main import run_main
    import sys
    sys.argv = ['', '--logdir='+self.logdir]
    sys.exit( run_main() )
