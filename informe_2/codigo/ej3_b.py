from knn import crossValidation
import numpy as np
import matplotlib.pyplot as plt
# Ahora queremos hacer 5 fold con n ks y quedarnos con el mejor basado en el promedio
# Para esto llamamos a la funcion efectividad k que dado un k hace el fold y devuelve el promedio

X_train = np.loadtxt("datos/X_train.csv", delimiter=",")
y_train = np.loadtxt("datos/y_train.csv", delimiter=",").astype(int)

# 1. Centramos los datos
media = np.mean(X_train, axis=0)

X_train = X_train - media

K = 300    # Cantidad de hiperparametros para los que queremos aplicar 5 fold cross-validation

datos = np.loadtxt("resultados/ej3_b")

# for i in range(len(datos)):
#     datos[i] = crossValidation(X_train, y_train, i+1)

# np.savetxt("resultados/ej3_b", datos)
plt.figure(0)
plt.plot(np.arange(1,K+1), datos)
plt.title("Exploración de hiperparámetro k para KNN")
plt.xlabel("# k")
plt.ylabel("Efectividad")
plt.savefig('graficos/ej3_b.png')