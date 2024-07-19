# Queremos ver cuánto tarda el método en converger para ciertos valores 
# Como las medidas son sobre datos aleatorios (para los autovectores) vamos a hacer 10 mediciones por autovalor y tomar la media y la varianza

import numpy as np
import matplotlib.pyplot as plt
from interfaz import obtenerDatos

cant_eps = 50
eps = np.logspace(-4,0,cant_eps)
cant_muestra = 20

iteraciones = np.empty((cant_eps,5, cant_muestra))  # Para cada epsilon luego tomaremos el promedio de iteraciones en converger
error = np.empty((cant_eps,5,cant_muestra))     # Error para cada lambda y cada vector aleatorio


for i in range(cant_eps): 
    autovalores = np.array([10, 10-eps[i], 5, 2, 1])
    D = np.diag(autovalores)

    # 1) Calculamos todos los autovalores y autovectores para cada caso
    for j in range(cant_muestra):     # Ahora por cada epsilon tomamos cant_muestra mediciones aleatorias
        v = np.random.rand(D.shape[0], 1)   # Creamos vectores aleatorios y los normalizamos
        v = v / np.linalg.norm(v)

        # Matriz de Householder
        B = np.eye(D.shape[0]) - 2 * (v @ v.T)

        M = B.T @ D @ B

        aval, avec, it = obtenerDatos(M,5)  # Obtenemos todos los autovalores, autovectores e iteraciones que tomó hallar cada autovalor

        for k in range(5):
            iteraciones[i][k][j] = it[k]
            dif = M@avec[:,k] - aval[k] * avec[:,k]
            error[i][k][j] = np.linalg.norm(dif)


promedio_it = np.empty((cant_eps,5))
promedio_error = np.empty((cant_eps,5))
desviacion_error = np.empty((cant_eps,5))    # Para cada lambda y epsilon habra una varianza en los resultados
desviacion_it = np.empty((cant_eps,5))    # Para cada lambda y epsilon habra una varianza en los resultados

for i in range(cant_eps):
    for j in range(5):
        promedio_error[i][j] = np.mean(error[i][j])
        desviacion_error[i][j] = np.std(error[i][j])
        promedio_it[i][j] = np.mean(iteraciones[i][j])
        desviacion_it[i][j] = np.std(iteraciones[i][j])


# Ahora con los promedios de iteraciones y errores podemos armar los gráficos
# Primero armamos la figura para los errores
plt.figure(0)
for i in range(5):
    plt.errorbar(eps, promedio_error[:,i],yerr=desviacion_error[:,i],fmt="o",  label=f'||A$v_{i}$ - $λ_{i}$$v_{i}$||$_{2}$')

plt.xlabel("Epsilon")
plt.ylabel("Error")
plt.yscale("log")
plt.xscale("log")
plt.legend(loc="upper right")
plt.grid()
plt.savefig("graficos/errores.png")

plt.figure(1)
for i in range(5):
    plt.errorbar(eps, promedio_it[:,i],yerr=desviacion_it[:,i], fmt="o",  label=f'$λ_{i}$')

plt.xlabel("Epsilon")
plt.ylabel("Iteraciones")
plt.yscale("log")
plt.xscale("log")
plt.grid()
plt.legend(loc="upper right")
plt.savefig("graficos/iteraciones.png")