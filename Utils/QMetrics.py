from itertools import product
import tensorflow as tf
import numpy as np

def tf_kron(a: tf.Tensor,
            b: tf.Tensor) -> tf.Tensor:

  """Calculates the Kronocker product of two matrices ((2,2) Tensors).

  Parameters
  -----------------------------------------------------------------------
  a(tf.Tensor): Tensor on the left of the product.
  b(tf.Tensor): Tensor on the right of the product.

  Return
  -----------------------------------------------------------------------
  (tf.tensor): Kronocker product between a and b.

  """
  assert len(a.shape) == 2
  assert len(b.shape) == 2
  a_shape = list(b.shape)    
  b_shape = list(b.shape)
  return tf.reshape(tf.reshape(a,[a_shape[0],1,a_shape[1],1])*tf.reshape(b,[1,b_shape[0],1,b_shape[1]]),[a_shape[0]*b_shape[0],a_shape[1]*b_shape[1]])


#Creating pauli matrices
sigma_0_np = np.array([[1.,0.],
                      [0.,1.]], dtype=np.complex64)

sigma_1_np = np.array([[0.,1.],
                       [1.,0.]], dtype=np.complex64)

sigma_2_np = np.array([[0.,1.j],
                       [-1.j,0.]], dtype=np.complex64)

sigma_3_np = np.array([[1.,0.],
                       [0.,-1.]], dtype=np.complex64)

#Converting to tensors
sigma_0 = tf.Variable(sigma_0_np, tf.complex64)

sigma_1 = tf.Variable(sigma_1_np, tf.complex64)

sigma_2 = tf.Variable(sigma_2_np, tf.complex64)

sigma_3 = tf.Variable(sigma_3_np, dtype=tf.complex64)

def create_2qubit_density_mat(measurements: tf.Variable) -> tf.Variable:
  
  name_basis_1 = ['I', 'X', 'Y', 'Z']
  basis_1 = [sigma_0, sigma_1,sigma_2,sigma_3]
  name_basis_2 = []
  basis_2 = []
  for (name_1, meas_1),(name_2,meas_2) in product(zip(name_basis_1, basis_1),zip(name_basis_1, basis_1)):
    if name_1 == 'I' and name_2 == 'I':
      pass
    else:
      basis_2.append(tf_kron(meas_1,meas_2))
      name_basis_2.append(name_1 + name_2)
  
  basis_2_tf = tf.Variable(basis_2, name='Basis')
  
  # Helper to make tr(density_matrix) = 1
  
  ones_II = tf.ones((tf.shape(measurements)[0],1), dtype=tf.dtypes.complex64)  
  II = tf.Variable([tf_kron(sigma_0 , sigma_0)], name='II')
  
  density_matrix = 0.25*(tf.tensordot(ones_II , II ,axes=1) + tf.tensordot(measurements,basis_2_tf,axes=1))
  return density_matrix

def trace_dist(A,B):
  dif = tf.math.subtract(A,B)  
  dif = tf.transpose(dif, conjugate=True, perm=[0,2,1]) * dif  
  vals = tf.linalg.eigvalsh(dif)
  return tf.math.real(0.5*tf.reduce_sum(tf.math.sqrt(tf.math.abs(vals)),axis=-1))

def trace_loss(y_true,y_pred):
  y_true = tf.cast(y_true, tf.dtypes.complex64, name='Casting_true')
  y_pred = tf.cast(y_pred, tf.dtypes.complex64, name='Casting_pred')
  d_y_true = create_2qubit_density_mat(y_true)  
  d_y_pred = create_2qubit_density_mat(y_pred)    
  return tf.reduce_mean(trace_dist(d_y_pred,d_y_true))

def entropy(A):
  vals_A = tf.linalg.eigvalsh(A)  
  return - tf.math.real(tf.reduce_sum(vals_A*tf.math.log(vals_A),axis=-1))

def relative_entropy(A,B):
  vals_A = tf.linalg.eigvalsh(A)
  vals_B = tf.linalg.eigvalsh(B)
  return - entropy(B) - tf.math.real(tf.reduce_sum(vals_A*tf.math.log(vals_B),axis=-1)) 

def q_cross_entropy(A,B):
  vals_A = tf.linalg.eigvalsh(A)
  vals_B = tf.linalg.eigvalsh(B)
  return - tf.math.real(tf.reduce_sum(vals_A*tf.math.log(vals_B),axis=-1)) 

def q_cross_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.dtypes.complex64, name='Casting_true')
  y_pred = tf.cast(y_pred, tf.dtypes.complex64, name='Casting_pred')
  d_y_true = create_2qubit_density_mat(y_true)  
  d_y_pred = create_2qubit_density_mat(y_pred)  
  return tf.reduce_mean(q_cross_entropy(d_y_pred,d_y_true))

def r_entropy_loss(y_true,y_pred):
  y_true = tf.cast(y_true, tf.dtypes.complex64, name='Casting_true')
  y_pred = tf.cast(y_pred, tf.dtypes.complex64, name='Casting_pred')
  d_y_true = create_2qubit_density_mat(y_true)  
  d_y_pred = create_2qubit_density_mat(y_pred)  
  return tf.reduce_mean(relative_entropy(d_y_pred,d_y_true))