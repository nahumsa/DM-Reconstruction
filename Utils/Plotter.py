#Helper functions from mitdeeplearning
import time
from IPython import display as ipythondisplay
from string import Formatter
import matplotlib.pyplot as plt

class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale
    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
      plt.cla()

      if self.scale is None:
        if len(data) == 2:
          plt.plot(data[0], label='Training')
          plt.plot(data[1], label='Validation')
          plt.legend()
        else:
          plt.plot(data)
        
      elif self.scale == 'semilogx':
        if len(data) == 2:
          plt.plot(data[0], label='Training')
          plt.plot(data[1], label='Validation')
          plt.legend()
        else:
          plt.plot(data)

      elif self.scale == 'semilogy':
        if len(data) == 2:
          plt.plot(data[0], label='Training')
          plt.plot(data[1], label='Validation')
          plt.legend()
        else:
          plt.plot(data)

      elif self.scale == 'loglog':
        if len(data) == 2:
          plt.plot(data[0], label='Training')
          plt.plot(data[1], label='Validation')
          plt.legend()
        else:
          plt.plot(data)
      
      else:
        raise ValueError("unrecognized parameter scale {}".format(self.scale))

      plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
      ipythondisplay.clear_output(wait=True)
      ipythondisplay.display(plt.gcf())

      self.tic = time.time()