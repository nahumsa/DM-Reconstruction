import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm.notebook import tqdm
from Utils.QMetrics import trace_loss, fidelity_rho
class Encoder(layers.Layer):
    """Maps Input to latent dimension."""

    def __init__(self,
                latent_dim,
                intermediate_dim,
                dropout_rate,
                **kwargs):
      
      super(Encoder, self).__init__(**kwargs)
      self.dense_proj = []
      for i in intermediate_dim:        
        self.dense_proj.append(layers.Dense(i,
                                            activation=tf.nn.relu))
      
      self.dropout_rate = dropout_rate
      
      if self.dropout_rate:
        self.dropout = layers.Dropout(self.dropout_rate)

      self.latent_layer = layers.Dense(latent_dim)    

    def call(self, inputs):
      x = self.dense_proj[0](inputs)
      
      for lay in self.dense_proj[1:]:
        x = lay(x)
        if self.dropout_rate:
          x = self.dropout(x)
          
      return self.latent_layer(x)

class Decoder(layers.Layer):
  """Decodes the encoded representation of the Encoder."""

  def __init__(self,
               original_dim,
               intermediate_dim,
               dropout_rate=None,
               **kwargs):
    super(Decoder, self).__init__(**kwargs)

    self.dense_proj = []
    for i in reversed(intermediate_dim):      
      self.dense_proj.append(layers.Dense(i,
                                          activation=tf.nn.relu))
    
    self.dropout_rate = dropout_rate
    
    if self.dropout_rate:
      self.dropout = layers.Dropout(self.dropout_rate)

    self.dense_output = layers.Dense(original_dim)

  def call(self, inputs):
    x = self.dense_proj[0](inputs)
    
    for lay in self.dense_proj[1:]:
      x = lay(x)
      if self.dropout_rate:
        x = self.dropout(x)

    return self.dense_output(x)
    
class TraceAutoEncoder(tf.keras.Model):
  """ AutoEncoder Model.

  Parameters
  -----------------------------------
  original_dim(int): Dimension of the input.
  intermediate_dim(list): Number of neurons on intermediate layers.
  latent_dim(int): Latent dimension.
  dropout_rate(float): Percentage of dropout.

  """

  def __init__(self,
               original_dim,
               intermediate_dim,
               latent_dim,
               dropout_rate,       
               **kwargs):
    super(TraceAutoEncoder, self).__init__()    
    self.encoder = Encoder(latent_dim=latent_dim, 
                           dropout_rate=dropout_rate,
                           intermediate_dim= intermediate_dim)
    self.decoder = Decoder(original_dim, 
                           dropout_rate= dropout_rate,
                           intermediate_dim= intermediate_dim)    
    self.fidelity = []
  
  def call(self, inputs):
    z = self.encoder(inputs)
    reconstructed = self.decoder(z)    
    return reconstructed

  def training_step(self, data, r_loss):
    """Training step for the AE.
  
    Parameters
    -------------------------------------------
    x: Data    
    optimizer(tf.keras.optimizer): Optimizer used.  
    r_loss(float): Parameter controlling reconstruction loss.    

    Return:
    Loss(float): Loss value of the training step.

    """
    x, y = data

    with tf.GradientTape() as tape:
      reconstructed = self(x)  # Compute input reconstruction.
      
      # Compute loss.
      loss = trace_loss(y, reconstructed)            
    
    # Update the weights of the VAE.
    grads = tape.gradient(loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))    
    
    fid = []
    for val_1, val_2 in zip(y.numpy(), reconstructed.numpy()):
      fid.append(fidelity_rho(val_1,val_2))    

    return loss, np.mean(fid)

  def validating_step(self, data, r_loss):
    """Validation step for the VAE.
  
    Parameters
    -------------------------------------------
    x: Data    
    r_loss(float): Parameter controlling reconstruction loss.    

    Return:
    Loss(float): Loss value of the training step.

    """
    x,y = data

    reconstructed = self(x)  # Compute input reconstruction.
    # Compute loss.
    loss = trace_loss(y, reconstructed)    
    
    fid = []
    for val_1, val_2 in zip(y.numpy(), reconstructed.numpy()):
      fid.append(fidelity_rho(val_1,val_2))    

    return loss, np.mean(fid)

  def training(self, dataset, 
             epochs, r_loss,              
             test= None ,Plotter=None):
    """ Training of the Variational Autoencoder for a 
    tensorflow.dataset.

    Parameters
    -------------------------------------------
    dataset(tf.data.Dataset): Dataset of the data.    
    epochs(int): Number of epochs.
    r_loss(float): Parameter controlling reconstruction loss.    
    Plotter(object): Plotter object to show how the training is
                    going (Default=None).

    """

    losses = []
    val_losses = []
    fidelities = []
    val_fidelities = []
    epochs = range(epochs)

    for i in tqdm(epochs, desc='Epochs'):
      losses_epochs = []
      fidelity_epochs =[]
      for step, x in enumerate(dataset):

        loss, fidelity = self.training_step(x, r_loss)
  
        # Logging.
        losses_epochs.append(float(loss))
        fidelity_epochs.append(float(fidelity))
      
      losses.append(np.mean(losses_epochs))
      fidelities.append(np.mean(fidelity_epochs))
      
      if test:
        val_losses_epochs = []
        val_fidelity_epochs = []

        for step, x in enumerate(test):

          val_loss, val_fidelity = self.validating_step(x, r_loss)
    
          # Logging.
          val_losses_epochs.append(float(val_loss))
          val_fidelity_epochs.append(float(val_fidelity))
        
        val_losses.append(np.mean(val_losses_epochs))
        val_fidelities.append(np.mean(val_fidelity_epochs))

      if Plotter != None:
        if test:
          Plotter.plot([losses,val_losses])          
        else:
          Plotter.plot(losses)
        

    return losses, val_losses, fidelities, val_fidelities