import matplotlib.pyplot as plt
import numpy as np
from ej2 import eliminacionGuassianaConPivoteoParcial

# Retorna el x calculado que mas se aleja de la solucion real para epsilon dado
def max_dif(sol):
    max: float = -1
    for i in range(3):
        dif: float = sol[i] - 1   # Restamos el x real
        dif = abs(dif)
        if dif > max:
            max = dif
    return max

epsilons = np.logspace(-6,0)
resultados32 = []
resultados64 = []

tolerancia = 1e-6   # Cual elegir?

for e in epsilons:

    a32 = np.array([[1, 2 + e, 3 - e], 
                 [1 - e, 2, 3 + e], 
                 [1+e, 2-e, 3]], dtype=np.float32)
    
    a64 = np.array([[1, 2 + e, 3 - e], 
                 [1 - e, 2, 3 + e], 
                 [1+e, 2-e, 3]], dtype=np.float64)
    
    b = np.array([6,6,6])

    # El epsilon de la funcion no tiene nada que ver con el de esta matriz (?)
    resultados32.append(max_dif(eliminacionGuassianaConPivoteoParcial(a32,b, tolerancia, dtype=32)))
    resultados64.append(max_dif(eliminacionGuassianaConPivoteoParcial(a64,b, tolerancia, dtype=64)))

    
# plt.hist(resultados32)
# plt.xlabel("Epsilons")
# plt.ylabel("Diferencia absoluta con solucion real")
# plt.show()
x = epsilons
y32 = np.array(resultados32)
y64 = np.array(resultados64)
plt.xscale('log')
plt.yscale('log')
plt.plot(x,y64, "o",label='Error para 64 bits')
plt.plot(x,y32,"o", label='Error para 32 bits')
plt.xlabel('Epsilons')
plt.ylabel('Error absoluto')
plt.legend()
plt.show()