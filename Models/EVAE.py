import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm.notebook import tqdm
from Utils.QMetrics import r_entropy_loss, q_cross_loss
from Models.TVAE import TraceVAE

class EntropyVAE(TraceVAE):
  def __init__(self,
               original_dim,
               intermediate_dim,
               latent_dim,               
               **kwargs):    

    super(EntropyVAE, self).__init__(original_dim,
                                   intermediate_dim,
                                   latent_dim)
    
  def training_step(self, x, r_loss, beta):
    """Training step for the VAE.
  
    Parameters
    -------------------------------------------
    x: Data.
    VAE(tf.keras.Model): Variational Autoencoder model. 
    optimizer(tf.keras.optimizer): Optimizer used.  
    r_loss(float): Parameter controlling reconstruction loss.
    beta(float): Parameter controlling the KL divergence.

    Return:
    Loss(float): Loss value of the training step.

    """
    with tf.GradientTape() as tape:
      reconstructed = self(x)  # Compute input reconstruction.
      # Compute loss.      
      loss = r_entropy_loss(x, reconstructed)
      kl = sum(self.losses)
      loss = r_loss * loss + beta*kl  
    
    # Update the weights of the VAE.
    grads = tape.gradient(loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))        
    return loss

class CrossEntropyVAE(TraceVAE):
  def __init__(self,
               original_dim,
               intermediate_dim,
               latent_dim,               
               **kwargs):    

    super(CrossEntropyVAE, self).__init__(original_dim,
                                   intermediate_dim,
                                   latent_dim)
    
  def training_step(self, x, r_loss, beta):
    """Training step for the VAE.
  
    Parameters
    -------------------------------------------
    x: Data.
    VAE(tf.keras.Model): Variational Autoencoder model. 
    optimizer(tf.keras.optimizer): Optimizer used.  
    r_loss(float): Parameter controlling reconstruction loss.
    beta(float): Parameter controlling the KL divergence.

    Return:
    Loss(float): Loss value of the training step.

    """
    with tf.GradientTape() as tape:
      reconstructed = self(x)  # Compute input reconstruction.
      # Compute loss.      
      loss = q_cross_loss(x, reconstructed)
      kl = sum(self.losses)
      loss = r_loss * loss + beta*kl  
    
    # Update the weights of the VAE.
    grads = tape.gradient(loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_weights))        
    return loss