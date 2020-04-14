from itertools import product
import qutip as qutip
import numpy as np

def measurement(density_matrix, base, name_base):
  """Measuring the quantum state on a given basis.
  """
  _measurements_names = []
  _measurements = []

  for (name_1, meas_1),(name_2,meas_2) in product(zip(name_base, base),zip(name_base, base)):
    #Ignore the II measurement because it is always 1
    if name_1 == 'I' and name_2 == 'I':
      pass
    else:
      measurement_op = qutip.tensor(meas_1,meas_2)
      _measurements.append(qutip.expect(measurement_op, density_matrix))
      _measurements_names.append(name_1 + name_2)

  return [i for i in zip(_measurements, _measurements_names)]

def DM_from_measurements(measurements, base, name_base):
  """Construct the Density Matrix from measurements.
  """
  
    #The order is the same as the measurements
  for i, ((name_1, basis_1),(name_2, basis_2)) in enumerate(product(zip(name_base, base),zip(name_base, base))):
      if name_1 == 'I' and name_2 == 'I':
          #Forcing trace = 1
          density_matrix = 1.*qutip.tensor(basis_1, basis_2)
      else: 
          density_matrix += measurements[i-1]*qutip.tensor(basis_1,basis_2)

  return density_matrix