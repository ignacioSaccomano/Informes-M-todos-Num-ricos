# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import mode


def calcular_exactitud(X_train, y_train, X_dev, y_dev, k):	# k = Cantidad de vecinos. El hiperparámetro del clasificador.
    
    cant_dev = len(y_dev)
    cant_test = len(y_train)
    exactitud = 0

    X_train /= np.linalg.norm(X_train, axis=1, keepdims=True)
    X_dev /= np.linalg.norm(X_dev, axis=1, keepdims=True)

    
    distancias = X_train @ X_dev.T     # X_train es t x 784 -> Ahora el producto T x D esta definido, y devuelve una matriz que en la columna j tiene las distancias del dato de dev j con cada dato de entrenamiento (fila i)
    

    distancias = np.ones((cant_test, cant_dev)) - distancias


    distancias = distancias.T   # Para tener en cada fila la cercania del dato de dev iesimo con los t datos de entrenamiento

    indices = np.argsort(distancias)
   
    etiquetas = y_train[indices]    # Con esto tenemos una matriz de igual dimension que la de indices pero con las etiquetas.

    vecinos = etiquetas[:, :k]  # Nos quedamos con las primeras k

    pred = mode(vecinos, axis=1)[0] # Ahora creamos un vector que tiene las etiquetas predichas para cada valor de acuerdo a la moda de los k vecinos.

    pred = pred.flatten()
    

    exactitud = np.sum((pred == y_dev).astype(int))


    return exactitud/cant_dev
    
        

def crossValidation(X_train, y_train, k):   # Pasamos el set de entrenamiento y vamos haciendo el folding

    resultados = np.empty(5)    # Aca guardamos los resultados de cada fold

    inicio = 0
    final = 1000
    for j in range(5):
        X_dev = X_train[inicio:final,:]
        X_newtrain = np.concatenate((X_train[:inicio, :], X_train[final:,:]), axis=0)
        y_dev = y_train[inicio:final]
        y_newtrain = np.concatenate([y_train[:inicio], y_train[final:]])
        resultados[j] = calcular_exactitud(X_newtrain, y_newtrain, X_dev, y_dev, k)
        inicio += 1000
        final += 1000


    return np.mean(resultados)   


def crossValidationPca(X, y,k, V):
    resultados = np.empty(5)
    inicio  = 0
    final = 1000
    for i in range(5):
        X_new = X @ V[i]
        X_dev = X_new[inicio:final,:]
        X_newtrain = np.concatenate((X_new[:inicio, :], X_new[final:,:]), axis=0)
        y_dev = y[inicio:final]
        y_newtrain = np.concatenate([y[:inicio], y[final:]])
        resultados[i] = calcular_exactitud(X_newtrain, y_newtrain, X_dev, y_dev, k)
        inicio += 1000
        final += 1000


    return np.mean(resultados)
