from Utils.Dataset import create_dataset_err
from Utils.Dataset import create_x
import numpy as np
import os 

name = "pi"
theta = np.pi
samples_train = 5000
measurements_train, measurements_errors_train = create_dataset_err(samples_train,
                                                                   theta=theta)

samples_test = 3000
measurements_test, measurements_errors_test = create_dataset_err(samples_test, 
                                                                 theta=theta)

#Transforming in an np.array
X_train = np.array(create_x(measurements_errors_train))
Y_train = np.array(create_x(measurements_train))

X_test = np.array(create_x(measurements_errors_test))
Y_test = np.array(create_x(measurements_test))

if not f'Data/Measurements_{name}/' in os.listdir(os.getcwd()):
    try:
        os.makedirs(f'Data/Measurements_{name}/') #create your directory
    except OSError as exc: 
        if exc.errno != errno.EEXIST:
            raise

np.savetxt(f'Data/Measurements_{name}/X_train.txt', X_train)
np.savetxt(f'Data/Measurements_{name}/Y_train.txt', Y_train)
np.savetxt(f'Data/Measurements_{name}/X_test.txt', X_test)
np.savetxt(f'Data/Measurements_{name}/Y_test.txt', Y_test)
