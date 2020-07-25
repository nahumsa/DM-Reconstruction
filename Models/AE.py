from tensorflow.keras import layers
import tensorflow as tf
import numpy as numpy

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

class AutoEncoder(tf.keras.Model):
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
    super(AutoEncoder, self).__init__()    
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

  def train_step(self, data):
    x, y = data 
    
    with tf.GradientTape() as tape:
      # Forward Pass
      y_pred = self(x, training=True)

      # Compute Loss Function
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    
    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    
    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}  

  def test_step(self, data):
    # Unpack the data
    x, y = data
    # Compute predictions
    y_pred = self(x, training=False)
    # Updates the metrics tracking the loss
    self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    # Update the metrics.
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).         
    return {m.name: m.result() for m in self.metrics}
