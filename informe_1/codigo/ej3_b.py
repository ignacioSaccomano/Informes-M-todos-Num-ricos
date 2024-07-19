import pytest
import numpy as np
import numpy.linalg as lng


# También conocido como algoritmo de Thomas
def eliminacionGaussianaST(a, b, c, d, dtype = 32):
    
    # Verifiquemos que los vectores tengan tamaño adecuado.
    n = len(b)
    if n != len(a)-1 != len(c)-1 != len(d):
        raise ValueError("Los vectores no tienen el tamaño adecuado.")
    
    if dtype == 32:
        tipo = np.single
    else:
        tipo = np.double
    # Usamos vectores de numpy
    a = np.array(a, tipo)
    b = np.array(b, tipo)
    c= np.array(c, tipo)
    d = np.array(d, tipo)
    

    # Modifica los coeficientes de la primera fila
    if b[0] == 0:
        raise ZeroDivisionError("El primer elemento de la diagonal principal es 0, por lo que no se puede aplicar el algoritmo directamente.")
    
    c[0] /= b[0]
    d[0] /= b[0]
    
    # En la mayoria de las implementaciones el indice de a arranca en 1 en lugar de 0. Para mantener coherencia agregamos un 0.
    a = np.concatenate(([0],a))
    
    for i in range(1, n ):
        temp = b[i] - (a[i] * c[i-1])
        if temp == 0:
            raise ZeroDivisionError("Un valor de b' es 0, por lo que no podemos calcular c' y d'.")
        
        if i != n - 1:
            c[i] /= temp 
        d[i] = (d[i] - (a[i] * d[i-1]))/temp
        
        

    # Sustitución hacia atrás
    res = np.zeros(n)
    res[n-1] = d[n - 1]

    for i in range(-2, -n - 1, -1):
        res[i] = d[i] - (c[i + 1] * res[i+1])

    #res = res.reshape(-1,1)
    return res

#########################################################################################################

# # Vectores de la matriz tridiagonal
# a = np.array([7, 2, 3])  # Diagonal inferior
# b = np.array([3, 4, 5, 6])  # Diagonal principal
# c = np.array([0, 1, 2])  # Diagonal superior

# # Vector de términos constantes
# d = np.array([7, 3, 4, 9])

# eliminacionGaussianaST(a,b,c,d)
# 1

"""
3a)  Derivar la formulación de EG para el caso tridiagonal.

    Venimos trabajando con el algoritmo de eliminación gaussiana para resolver sistemas de ecuaciones lineales.
Ya sea que estamos en el caso con o sin pivoteo, los pasos se resumen en armar la matriz ampliada, convertirla en una triangular
superior a través de operaciones de resta y multiplicación por escalar y por último resolver el sistema por sustitución u 
otro método. Este algoritmo es el más efectivo cuando trabajamos con matrices sin ninguna característica en particular y permite 
resolver el sistema en una complejidad de O(n^3).
    Ahora nos toca analizar el caso para matrices tridiagonales, las cuales se caracterizan por tener ceros por fuera de la 
diagonal principal, inferior y superior. A simple vista se ve que muchas de las operaciones que lleva a cabo la EG serán anuladas
por la existencia de los ceros. Con leves cambios se puede conseguir un algoritmo más eficiente, como el algoritmo de Thomas. 
A continuación, veremos que aplicando EG podremos construir fórmulas que resumen los cambios de las variables.
    Sea A una matriz tridiagonal de tamaño n cuya diagonal inferior se representa por el vector a con índices de 2 a n, su diagonal 
principal se representa por el vector b con índices de 1 a n y su diagonal superior se representa por el vector c con índices de 1 a 
n-1. Sea d el vector que tomara el papel de b en la resolución de Ax = b, veamos que tenemos:

    Tomamos n = 4 pero se puede aplicar para cualquier n
    
    [ b1, c1, 0, 0]  [x1]        [d1]
    [ a2, b2, c2, 0] [x2]        [d2]
    [ 0, a3, b3, c3] [x3]    =   [d3]
    [ 0, 0, a4, b4]  [x4]        [d4]
    
    
    Iteramos en orden por todas las filas i en la matriz ampliada y hacemos:
    
        Primero dividimos por bi. En el caso i = 1:
        a1 no existe
        b1 = b1 / b1 = 1 
        c1 = c1 / b1 (Como este será el valor final de c1, la llamaremos c1')
        d1 = d1 /b1 (Como este será el valor final de d1, la llamáramos d1').
        
        Luego le restamos a la fila i +1 la i multiplicada por a i+1. En el caso de i = 1: 
        a2 = a2 - (b1 * a2) = 0
        b2 = b2 - (c1' * a2)
        c2 = c2 - (0 * a2) = c2
        d2 = d2 - (d1' * a2)
        
    Cuando pasemos a la fila 2, haremos los mismo 2 pasos y tendremos:
    
        i = 2
        
        Dividimos por b2, que en este caso ya cambio y es b2 = b2 - (c1' * a2):
        a2 ya era 0
        b2 = 1
        c2 = c2 / b2 - (c1' * a2) (Como este será el valor final de c2, la llamaremos c2')
        d2 = d2 - (d1' * a2) / b2 - (c1' * a2) (Como este será el valor final de d2, la llamaremos d2')
        
        Restamos a la fila 3 la 2 multiplicada por a3:
        
        ...
        
    Notemos que los únicos coeficientes importantes ci' y di' serán de la forma:
    
    
    c' = ci / bi para i = 1 
        = ci / (bi - ai * ci-1') para i de 2 a n-1
    
    
    d' = di / bi para i = 1
        = (di - ai * d i-1') / (bi - ai * c i-1') para i de 2 a n
        
    Basándonos en estas fórmulas podemos crear un algoritmo de complejidad temporal 0(n) que primero 
    calcule estos coeficientes de 1 a n y luego realice la dicha sustitución.


"""