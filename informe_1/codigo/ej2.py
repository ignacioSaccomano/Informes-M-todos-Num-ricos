import pytest
import numpy as np
import numpy.linalg as lng
from kahan import kahan
import random

def cambioDeFila(matriz, fila1, fila2):
    matriz[[fila1, fila2]] = matriz[[fila2, fila1]]
    return matriz

def eliminacionGuassianaConPivoteoParcial ( m , b, tolerancia, dtype=32):
    
    if len(m) != len(m[0]):# Verificamos que sea cuadrada
        raise ValueError("La matriz ingresada no es cuadrada.")
    
    m = np.array(m)
    b = np.array(b)

    m = np.column_stack([m,b]) #  Crea la matriz aumentada (para no tratar a b y m aparte)
    if dtype == 32:
        m.astype(np.single)
    else:
        m.astype(np.double)

    n : int = len(m)

    # Primero debemos transformar a nuestra matriz cuadrada en una matriz triangular superior

    for i in range(0, n - 1):
        
        max_fila = i
        for s in range(i, n):
            if np.abs(m[s][i]) > np.abs(m[max_fila][i]):
                max_fila = s
        
        if m[max_fila][i] == 0:# En este caso todos los valores por debajo del 0 en la diagonal son 0, por lo que salteamos el paso.
            continue
            
        elif max_fila != i: # Es decir, el maximo NO es nuetro pivote actual, necesitamos un cambio.
            cambioDeFila( m, i, max_fila)
        
        
         
        for j in range(i + 1, n):
            
            if m[i][i] != 0:
                if(np.abs(m[i][i])<tolerancia):
                    print("ADVERTENCIA: El dividendo es menor a la tolerancia, se acerca mucho a cero.")
                
            temp: float = m[j][i] / m[i][i]
    
            for k in range(i, n +1):
                m[j][k] = m[j][k] - (temp * m[i][k])


    # Una vez obtenemos la matriz diagonal superior, toca verificar si el sistema es compatible indeterminado o incompatible. A efectos de nuestro problema, sea cual sea el caso devolvemos error porque no existe una solucion unica.
    if m[n-1][n-1] == 0:
        if m[n-1][n] ==0:
            raise ValueError("El sistema es compatible indeterminado.")
        else:
            raise ValueError("El sistema es incompatible.")
        
    # Resolvemos el sistema.
        
    res = np.zeros(n)
    for i in range(n - 1, -1, -1):
        res[i] = (m[i][n] - kahan([m[i][j] * res[j] for j in range(i + 1, n)])) / m[i][i]

    res = res.reshape(-1,1)
    return res

"""
#########################################################################################################
#2-b) sistema compatible indeterminado
# Matriz de coeficientes A y vector de términos constantes b para un sistema compatible indeterminado
A_indeterminado = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [0, 0, 0]  # Una fila de ceros
])

b_indeterminado = np.array([1, 2, 0])  # b tiene un término no nulo correspondiente a una fila de ceros en A

#print(f"Si planteamos una matriz que sea complatible indeterminada, tomamos a la matriz: \n {A_indeterminado} \n con el vector solucion y : \n {b_indeterminado} \n y ademas una tolerancia de {tolerancia}, \n si aplicamos la funcion 'eliminacionGuassianaConPivoteoParcial' obtendremos : \n {eliminacionGuassianaConPivoteoParcial(A_indeterminado,b_indeterminado,tolerancia)}")



# Sistema incompatible
# Matriz de coeficientes A y vector de términos constantes b para un sistema incompatible
A_incompatible = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [0, 0, 0]  # Una fila de ceros
])

b_incompatible = np.array([1, 2, 1])  # b tiene un término no nulo correspondiente a una fila de ceros en A

#########################################################################################################
#2-c)

tolerancia = 1e-5

error = 1e-6
numeroAleatorio = random.random()
#creamos una matriz de 32bits


# Defino una matriz cuadrada de tipo float32 (32 bits)
A_float32 = np.array([[1.0, 2.0, 3.0+error],
                      [4.0+error, 5.0, 6.0-error],
                      [7.0, 8.0+error, 9.0-error]], dtype=np.float32)

# Defino un vector columna (por ejemplo, [1, 2, 3]) de tipo float32
x_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(-1, 1)  # Convertir a columna

# Calculo el vector imagen y = A_float32 @ x_float32
vector_imagen_float32 = np.dot(A_float32, x_float32)

#ahora calculo el vector solucion que me da si aplico el algoritmo
x_moño32 = eliminacionGuassianaConPivoteoParcial(A_float32,vector_imagen_float32,tolerancia)
print("2-c")
print("ahora los datos de 32bits")
print(f"si comparamos la solucion original sin error que es: \n {x_float32} \n con el que nos da la implementacionque es: \n {x_moño32}")
print(f"la diferencia entre los dos vectores es: \n {x_float32 - x_moño32} \n la norma infinito de la diferencia entre los dos vectores es: {np.linalg.norm(x_float32 - x_moño32,ord=np.inf)}")



#################
# Defin una matriz cuadrada de tipo float64 (64 bits)
A_float64 = np.array([[1.0+error, 2.0-error, 3.0],
                      [4.0, 5.0+error, 6.0-error],
                      [7.0+error, 8.0, 10.0-error]], dtype=np.float64)

# Defino un vector columna (por ejemplo, [1, 2, 3]) de tipo float64
x_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(-1, 1)  # Convertir a columna

# Calculo el vector imagen y = A_float64 @ x_float64
vector_imagen_float64 = np.dot(A_float64, x_float64)
print("ahora los datos de 64bits")
x_moño64 = eliminacionGuassianaConPivoteoParcial(A_float64,vector_imagen_float64,tolerancia)
"""