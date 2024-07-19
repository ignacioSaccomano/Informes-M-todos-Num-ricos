from pca import pca
import matplotlib.pyplot as plt
import numpy as np

X_train = np.loadtxt("datos/X_train.csv", delimiter=",")

# 1. Centramos los datos
media = X_train.mean(axis=0)

X_train = X_train - media

# varianzas,_ = pca(X_train, 784)
varianzas = np.loadtxt("resultados/autovalores_final.txt")

varianzas = varianzas / np.sum(varianzas)

plt.figure(0)
plt.plot(np.cumsum(varianzas))
plt.grid()
plt.ylabel("Varianza explicada acumulada")
plt.xlabel('# componentes')
plt.savefig("graficos/ej3_c.png")

plt.figure(1)
plt.plot(np.arange(1,785), varianzas)
plt.ylabel("Varianza explicada")
plt.xlabel('# componentes')
plt.yscale("log")
plt.title("Varianza explicada por todas las componentes")
plt.savefig("graficos/varianzas_separado.png")

plt.figure(2)
plt.plot(np.arange(1,101), varianzas[:100])
plt.ylabel("Varianza explicada")
plt.xlabel('# componentes')
plt.title("Varianza explicada por las primeras 100 componentes")
plt.savefig("graficos/varianzas_separado_zoom.png")
