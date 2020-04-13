from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, Concatenate
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model
import tensorflow as tf

import numpy as np
import json
import os
import pickle
      
        
class DenseVariationalAutoencoderKeras():
    """
    Variational Autoencoder with Dense layers.
    
    In order to acces the encoder use self.encoder, the same for the decoder.
    If want to acces the VAE, use model.
    
    Parameters
    -------------------------------------------------------------------------
    input_dim(tuple): Dimentions of the input.
    encoder_dense_units(list): Units of the dense layer of the encoder.
    decoder_dense_units(list): Units of the dense layer of the decoder.
    z_dim(int): Dimension of the latent layer
    use_batch_norm(Boolean): True if want to use batch normalization.
                             (default=False)
    use_dropout(Boolean): True if want to use dropout layers.
                             (default=False)
    
    """
    def __init__(self
        , input_dim
        , encoder_dense_units
        , decoder_dense_units
        , z_dim
        , use_batch_norm = False
        , use_dropout= False
        ):

        self.name = 'variational_autoencoder_Dense'
        
        self.input_dim = input_dim
        
        self.encoder_dense_units = encoder_dense_units
        
        self.decoder_dense_units = decoder_dense_units
        
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_dense_units)
        self.n_layers_decoder = len(decoder_dense_units)

        self._build()

    def _build(self):
        
        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        
        x = encoder_input

        for i in range(self.n_layers_encoder):
            
            dense_layer = Dense( 
                self.encoder_dense_units[i]
                , name = 'encoder_dense_' + str(i)
                )
            
            x = dense_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate = 0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]
        
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)
        
        

        ### THE DECODER

        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')

        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            
            
            dense_layer = Dense( 
                self.decoder_dense_units[i]
                , name = 'decoder_dense_' + str(i)
                )

            x = dense_layer(x)

            if i < self.n_layers_decoder - 1: 
                
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                
                x = LeakyReLU()(x)
                
                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('linear')(x)

            

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)


    def compile(self, learning_rate, r_loss_factor, Beta):
        """
        Compiling the network. Need to choose the learning rate, r_loss_factor
        and Beta for the Beta-VAE, if Beta = 1 then it is a VAE.
        
        Parameters
        ----------------------------------------------------------------------
        learning_rate: Learning Rate for gradient descent.
        r_loss_factor: Factor that multiplies the loss factor of the
                       reconstruction loss.
        Beta: Beta-VAE parameter that multiplies the KL-Divergence in order to
              disentangle the latent space of the model.
              
        """
        
        self.learning_rate = learning_rate        

        ### COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = -1)
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + Beta*kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, 
                           loss = vae_loss,  
                           metrics = ['accuracy', 
                                      vae_r_loss, 
                                      vae_kl_loss]
                          )


    def save(self, folder):
        """ Save the model's parameters 
        on a pickle file.
        
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.encoder_dense_units
                , self.decoder_dense_units
                , self.z_dim
                , self.use_batch_norm 
                , self.use_dropout
                ], f)

        self.plot_model(folder)


    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, y_train, 
                    batch_size, validation_data, 
                    epochs, run_folder, verbose=2, 
                    print_every_n_batches = 100, 
                    initial_epoch = 0, callbacks = None):
        """
        Arguments:
            x_train {np.array} -- Train X.
            y_train {np.array} -- Train Y
            batch_size {int} -- Batch Size.
            validation_data {tuple} -- Tuple of validation data.
                                       Ex: (test_X, test_Y)
            epochs {int} -- Number of epochs.
            run_folder {string} -- Folder to run/save model.
        
        Keyword Arguments:
            verbose {int} -- Verbose of training. (default: {2})
            print_every_n_batches {int} -- Number of batches that you want to print
                                           results. (default: {100})
            initial_epoch {int} -- Starting epoch (default: {0})
            callbacks {list} -- Callbacks of the training (default: {None})
            
        """
                
        checkpoint_filepath=os.path.join(run_folder, 
                                         "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, 
                                      save_weights_only = True, 
                                      verbose=0)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 
                                                   'weights/weights.h5'), 
                                      save_weights_only = True, 
                                      verbose=0)

        callbacks_list = [checkpoint1, checkpoint2]
        
        # Check if there are new callbacks
        if callbacks != None:
          for call in callbacks:
            callbacks_list.append(call)
        
        history = self.model.fit(     
                                  x_train
                                  , y_train
                                  , batch_size = batch_size
                                  , shuffle = True
                                  , epochs = epochs
                                  , verbose = verbose
                                  , initial_epoch = initial_epoch
                                  , callbacks = callbacks_list
                                  , validation_data = validation_data
                                 )
        return history

        
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)

        
class ScinetVariationalAutoencoderKeras():
    """
    Variational Autoencoder with Dense layers.
    
    In order to acces the encoder use self.encoder, the same for the decoder.
    If want to acces the VAE, use model.
    
    Parameters
    -------------------------------------------------------------------------
    input_dim(tuple): Dimentions of the input.
    encoder_dense_units(list): Units of the dense layer of the encoder.
    decoder_dense_units(list): Units of the dense layer of the decoder.
    z_dim(int): Dimension of the latent layer.
    q_dim(int): Dimension of the question layer.
    use_batch_norm(Boolean): True if want to use batch normalization.
                             (default=False)
    use_dropout(Boolean): True if want to use dropout layers.
                             (default=False)
    
    """
    def __init__(self
        , input_dim
        , encoder_dense_units
        , decoder_dense_units
        , z_dim
        , q_dim
        , use_batch_norm = False
        , use_dropout= False
        ):

        self.name = 'variational_autoencoder_Dense'
        
        self.input_dim = input_dim
        
        self.encoder_dense_units = encoder_dense_units
        
        self.decoder_dense_units = decoder_dense_units
        
        self.z_dim = z_dim
        self.q_dim = q_dim
        
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_dense_units)
        self.n_layers_decoder = len(decoder_dense_units)

        self._build()

    def _build(self):
        
        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            
            dense_layer = Dense( 
                self.encoder_dense_units[i]
                , name = 'encoder_dense_' + str(i)
                )
            
            x = dense_layer(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.use_dropout:
                x = Dropout(rate = 0.25)(x)

        shape_before_flattening = K.int_shape(x)[1:]                
        
        self.mu = Dense(self.z_dim, name='mu')(x)
        self.log_var = Dense(self.z_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))

        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon

        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)
        
        

        ### THE DECODER                
        
        question_input = Input(shape=self.q_dim, name='question_input') 
        
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        
        Merge = Concatenate(axis=1, name='Concatenate_Q_Z')([question_input,decoder_input])                
        
        x = Dense(np.prod(shape_before_flattening))(Merge)
        

        for i in range(self.n_layers_decoder):
            
            
            dense_layer = Dense( 
                self.decoder_dense_units[i]
                , name = 'decoder_dense_' + str(i)
                )

            x = dense_layer(x)

            if i < self.n_layers_decoder - 1: 
                
                if self.use_batch_norm:
                    x = BatchNormalization()(x)
                
                x = LeakyReLU()(x)
                
                if self.use_dropout:
                    x = Dropout(rate = 0.25)(x)
            else:
                x = Activation('sigmoid')(x)

            

        decoder_output = x

        self.decoder = Model([question_input,decoder_input], decoder_output)

        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder([question_input,encoder_output])

        self.model = Model([model_input,question_input], model_output)


    def compile(self, learning_rate, r_loss_factor, Beta):
        """
        Compiling the network. Need to choose the learning rate, r_loss_factor
        and Beta for the Beta-VAE, if Beta = 1 then it is a VAE.
        
        Parameters
        ----------------------------------------------------------------------
        learning_rate: Learning Rate for gradient descent.
        r_loss_factor: Factor that multiplies the loss factor of the
                       reconstruction loss.
        Beta: Beta-VAE parameter that multiplies the KL-Divergence in order to
              disentangle the latent space of the model.
              
        """
        
        self.learning_rate = learning_rate

        ### COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = -1)
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + Beta*kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])


    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.encoder_dense_units
                , self.decoder_dense_units
                , self.z_dim
                , self.use_batch_norm 
                , self.use_dropout
                ], f)

        self.plot_model(folder)


    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, y_train, batch_size, 
                    epochs, run_folder, verbose=2, 
                    print_every_n_batches = 100, 
                    initial_epoch = 0, lr_decay = 1):

                
        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)

        callbacks_list = [checkpoint1, checkpoint2]

        self.model.fit(     
            x_train
            , y_train
            , batch_size = batch_size
            , shuffle = True
            , epochs = epochs
            , verbose = verbose
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
        )



    def train_with_generator(self, data_flow, epochs, steps_per_epoch, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1, ):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint_filepath=os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = ModelCheckpoint(checkpoint_filepath, save_weights_only = True, verbose=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)

        callbacks_list = [checkpoint1, checkpoint2, custom_callback, lr_sched]

        self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                
        self.model.fit_generator(
            data_flow
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callbacks_list
            , steps_per_epoch=steps_per_epoch 
            )


    
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)
