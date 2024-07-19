import numpy as np
import numpy.linalg as lng
import timeit
import matplotlib.pyplot as plt
from ej2 import eliminacionGuassianaConPivoteoParcial
from ej3_b import eliminacionGaussianaST

def llamadora_tri():
    
    eliminacionGaussianaST(a,b,c, d, 64)
    
def llamadora_con():
    
    eliminacionGuassianaConPivoteoParcial(mLaplaciana, d, 1e-6, 64)

tamaños = np.arange(3,20)
tiempos_TRI = []
tiempos_CON = []

for i in range(len(tamaños)):
    
    n = tamaños[i]

    # Creamos la matriz laplaciana

    a = np.full( n - 1, 1, np.double)
    b = np.full( n, -2, np.double)
    c = np.full( n -1, 1, np.double)
    
    mLaplaciana = np.diag(b) + np.diag(c, 1) + np.diag(a, -1)
    mLaplaciana = mLaplaciana.astype(np.double)
    
    # Que valor de d debemos usar? Por ahora usamos uno aleatorio
    d = np.zeros(n, dtype=np.float64)
    

    # Ejecuta la expresión varias veces y obtén los tiempos de ejecución
    time_SIN = timeit.repeat( llamadora_tri, repeat=5, number=100)
    time_CON = timeit.repeat(llamadora_con, repeat=5, number= 100)
    mejor_tiempo_TRI = min(time_SIN)
    mejor_tiempo_CON = min(time_CON)
    
    tiempos_TRI.append(mejor_tiempo_TRI)
    tiempos_CON.append(mejor_tiempo_CON)
    

# x va a tomar el rol del tamaño de las matrices, mientras que y el mejor tiempo promedio
    
    
plt.plot( tamaños,tiempos_TRI, "-",label='EG sistema tridiagonal')
plt.plot( tamaños,tiempos_CON, "-",label='EG con pivoteo parcial')

plt.xlabel('Tamaño')
plt.ylabel('Tiempo')
plt.xscale('log')
plt.yscale('log')
plt.title("EG con pivoteo vs solución tridiagonal")
plt.legend(loc='upper left')
plt.show()