from knn import calcular_exactitud
import numpy as np

X_train = np.loadtxt("datos/X_train.csv", delimiter=",")
y_train = np.loadtxt("datos/y_train.csv", delimiter=",").astype(int)
X_test = np.loadtxt("datos/X_test.csv", delimiter=",")
y_test = np.loadtxt("datos/y_test.csv", delimiter=",").astype(int)


# Centramos los datos por como definimos la funcion de correlacion
media_train = X_train.mean(axis=0)
media_test =  X_test.mean(axis=0)
X_train = X_train - media_train
X_test = X_test - media_test


print(calcular_exactitud(X_train, y_train, X_test, y_test,5))