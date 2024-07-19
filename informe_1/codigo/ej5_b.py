import numpy as np
import numpy.linalg as lng
import timeit
import matplotlib.pyplot as plt
from ej1 import eliminacionGuassianaSinPivoteo
from ej2 import eliminacionGuassianaConPivoteoParcial
from ej3_b import eliminacionGaussianaST
from ej3_c import st_precomputo, auxiliar_precomputo

def llamadora_NOPRE():
    
    eliminacionGaussianaST( a, b, c, d, 64)
    
def llamadora_PRE():
    
    st_precomputo( C, den, a,d,64)



enes = np.arange(1,100)
tiempos_NOPRE = []
tiempos_PRE = []

# No nos piden un tamaño en especifico.
tamaño = 10


    
    
for n in enes:
    
    a = np.full(tamaño - 1, 1, np.double)
    b = np.full(tamaño, -2, np.double)
    c = np.full(tamaño - 1, 1, np.double)
    den, C = auxiliar_precomputo(a, b, c, 64)
    
    mejor_tiempo_NOPRE = 0.0
    mejor_tiempo_PRE = 0.0
    
    for _ in range(n):
        
        d = np.zeros(tamaño, dtype=np.double)
        
        time_NOPRE = timeit.repeat(llamadora_NOPRE, repeat=5, number=100)
        time_PRE = timeit.repeat(llamadora_PRE, repeat=5, number=100)
        
        mejor_tiempo_NOPRE += min(time_NOPRE)
        mejor_tiempo_PRE += min(time_PRE)
    
    
    tiempos_NOPRE.append(mejor_tiempo_NOPRE)
    tiempos_PRE.append(mejor_tiempo_PRE)
    

# En este caso, x va a tomar el rol de n, siendo este la cantidad de veces que se evalua un mismo sistema y y va a ser el tiempo. Recordemos que 
# supuestamente preecomputar ciertos valores reduce el tiempo de computo total.

plt.plot( enes,tiempos_NOPRE, "-",label='sin precomputo')
plt.plot( enes,tiempos_PRE, "-",label='con precomputo')


plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('tiempo')
plt.legend()
plt.show()
    
    
    
    
    
    

