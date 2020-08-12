from itertools import product
import qutip as qutip
import numpy as np

from Utils.QutipUtils import measurement, add_error

def create_dataset(n_samples):
  """Create dataset.
  
  Parameters:
  n_samples(int): Number of samples
  
  Output:
  _states(list): States associated with each set of measurements.
  _measurements(list): Measurements.
  _labels(list): Label associated with each set of measurements.
  """

  _states = []
  _labels = []
  _measurements = []

  #Basis Measured
  name_basis = ['I', 'X', 'Y', 'Z']
  basis = [qutip.identity(2), qutip.sigmax(),qutip.sigmay(),qutip.sigmaz()]



  for _ in range(n_samples):    
    density = qutip.rand_dm(4, density=0.75, dims=[[2,2],[2,2]])
    
    #Partial Transpose
    density_partial_T = qutip.partial_transpose(density, [0,1])    
  
    #Labels: 1 if entangled 0 if separable (PPT Criterion)
    if (density_partial_T.eigenenergies() < 0).any():
      _labels.append(1)
  
    else:      
      _labels.append(-1)  

    _states.append(density)  
  
    val_measurements = measurement(density_matrix=density, 
                                   base=basis, 
                                   name_base=name_basis)
  
    _measurements.append(val_measurements)
    
  return _states, _measurements, _labels

def create_dataset_err(n_samples, theta=0., phi=0., lbda=0., qubit=1):
  """Create dataset.
  
  Parameters:
  n_samples(int): Number of samples.
  theta(float): Parameter for the error.
  phi(float): Parameter for the error.
  lbda(float): Parameter for the error.
  qubit(int): In which qubit you want to put the error.
  
  Output:
  _measurements(list): Measurements.
  _measurements_err(list): Measurements with error.
  """
  
  _measurements = []
  _measurements_err = []

  #Basis Measured
  name_basis = ['I', 'X', 'Y', 'Z']
  basis = [qutip.identity(2), qutip.sigmax(),qutip.sigmay(),qutip.sigmaz()]



  for _ in range(n_samples):    
    density = qutip.rand_dm(4, density=0.75, dims=[[2,2],[2,2]])
    
    theta = theta*np.random.randn()
    
    density_err = add_error(density, theta, phi, lbda, qubit)    
  
    val_measurements = measurement(density_matrix=density, 
                                   base=basis, 
                                   name_base=name_basis)
    
    val_measurements_err = measurement(density_matrix=density_err,
                                       base=basis,
                                       name_base=name_basis)        
    
    _measurements.append(val_measurements)
    _measurements_err.append(val_measurements_err)
    
  return _measurements, _measurements_err

#Unpacking the training data
def create_x(measurement):
  """Create an list with all measurements
  Parameters:
  measurement(list): List of measurements and the basis measured.  

  """
  X = []
  for meas in measurement:
    aux = []
    for result , name in meas:      
      aux.append(result)
    X.append(aux)
  return X

def create_x_correlated(measurement):
  """Create an list with correlated measurements
  Parameters:
  measurement(list): List of measurements and the basis measured.  

  """
  X = []
  for meas in measurement:
    aux = []
    for result , name in meas:
      if name[0] == 'I' or name[1] == 'I':  
        pass
      else:
        aux.append(result)
    X.append(aux)
  return X

def create_x_local(measurement):
  """Create an list with local measurements
  Parameters:
  measurement(list): List of measurements and the basis measured.
  
  """
  X = []
  for meas in measurement:
    aux = []
    for result , name in meas:
      if name[0] == 'I' or name[1] == 'I':  
        aux.append(result)        
    X.append(aux)
  return X

def generate_separable(n_samples):
  """Create n_samples separable states.
  
  Parameters:
  n_samples(int): Number of samples
  
  Output:
  states_train(list): States associated with each set of measurements.
  measurements_train(list):  measurements
  """
  _states = []  
  _measurements = []

  #Basis Measured
  name_basis = ['I', 'X', 'Y', 'Z']
  basis = [qutip.identity(2), qutip.sigmax(),qutip.sigmay(),qutip.sigmaz()]


  counter = 0

  while not counter == n_samples:    
    density = qutip.rand_dm(4, density=0.75, dims=[[2,2],[2,2]])
    
    #Partial Transpose
    density_partial_T = qutip.partial_transpose(density, [0,1])    
  
    #Labels: 1 if entangled 0 if separable (PPT Criterion)
    if (density_partial_T.eigenenergies() < 0).any():      
      pass
  
    else:      
      _states.append(density)  
  
      val_measurements = measurement(density_matrix=density, 
                                   base=basis, 
                                   name_base=name_basis)
  
      _measurements.append(val_measurements)
      
      counter += 1

  return _states, _measurements

def generate_entangled(n_samples):
  """Create n_samples entangled states.
  
  Parameters:
  n_samples(int): Number of samples
  
  Output:
  states_train(list): States associated with each set of measurements.
  measurements_train(list):  measurements
  """
  _states = []  
  _measurements = []

  #Basis Measured
  name_basis = ['I', 'X', 'Y', 'Z']
  basis = [qutip.identity(2), qutip.sigmax(),qutip.sigmay(),qutip.sigmaz()]


  counter = 0

  while not counter == n_samples:    
    density = qutip.rand_dm(4, density=0.75, dims=[[2,2],[2,2]])
    
    #Partial Transpose
    density_partial_T = qutip.partial_transpose(density, [0,1])    
  
    #Labels: 1 if entangled 0 if separable (PPT Criterion)
    if (density_partial_T.eigenenergies() < 0).any():      
      _states.append(density)  
  
      val_measurements = measurement(density_matrix=density, 
                                   base=basis, 
                                   name_base=name_basis)
  
      _measurements.append(val_measurements)
      
      counter += 1            

  return _states, _measurements