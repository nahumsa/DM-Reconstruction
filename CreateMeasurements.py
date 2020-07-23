from Utils.Dataset import create_dataset
from Utils.Dataset import create_x

samples_train = 5000
states_train, measurements_train, labels_train = create_dataset(samples_train)

samples_test = 3000
states_test, measurements_test, labels_test = create_dataset(samples_test)

#Transforming in an np.array
X_train = np.array(create_x(measurements_train))

X_test = np.array(create_x(measurements_test))

np.savetxt('X_train.txt', X_train)
np.savetxt('X_test.txt', X_test)
