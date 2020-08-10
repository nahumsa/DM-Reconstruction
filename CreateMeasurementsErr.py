from Utils.Dataset import create_dataset_err
from Utils.Dataset import create_x

samples_train = 5000
measurements_train, measurements_errors_train = create_dataset_err(samples_train)

samples_test = 3000
measurements_test, measurements_errors_test = create_dataset_err(samples_test)

#Transforming in an np.array
X_train = np.array(create_x(measurements_errors_train))
Y_train = np.array(create_x(measurements_train))

X_train = np.array(create_x(measurements_errors_test))
Y_train = np.array(create_x(measurements_test))

np.savetxt('Data/Errors/X_train.txt', X_train)
np.savetxt('Data/Errors/Y_train.txt', Y_train)
np.savetxt('Data/Errors/X_test.txt', X_test)
np.savetxt('Data/Errors/Y_test.txt', Y_test)
