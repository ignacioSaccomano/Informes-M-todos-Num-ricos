import pytest
import numpy as np
import numpy.linalg as lng

"""
Porque nos interesaría precomputar las actualizaciones que sufren a, b y c? Para utilizarlas con distintos d. Veamos las fórmulas que utilizamos anteriormente:

c'  = ci / bi para i = 1 
    = ci / (bi - ai * ci-1') para i de 2 a n-1
    
    
d'  = di / bi para i = 1
    = (di - ai * d i-1') / (bi - ai * c i-1') para i de 2 a n
 
Todos los coeficientes de c' dependen únicamente de a, b y c, no de d. El denominador de d' (bi - ai * c i-1') tampoco dependen de d. Precomputando estos valores
podríamos utilizarlos para resolver los sistemas de ecuaciones que compartan a, b y c, o mejor dicho la matriz tridiagonal. 

Al precomputo de c' lo llamaremos C y al precomputo de (bi - ai * c i-1') lo llamaremos den (denominador).

"""
def auxiliar_precomputo( a, b, c, dtype = 32):
    
    # Verifiquemos que los vectores tengan tamaño adecuado.
    n = len(b)
    if n != len(a)-1 != len(c)-1:
        raise ValueError("Los vectores no tienen el tamaño adecuado.")
    
    if dtype == 32:
        tipo = np.single
    else:
        tipo = np.double
    # Usamos vectores de numpy
    a = np.array(a, tipo)
    b = np.array(b, tipo)
    c= np.array(c, tipo)
    
    
    # En la mayoria de las implementaciones el indice de a arranca en 1 en lugar de 0. Para mantener coherencia agregamos un 0.
    a = np.concatenate(([0.0],a))
    
    
    C = np.zeros(n-1)
    den = np.zeros(n)
    
    # Caso i = 1
    den[0] = b[0]
    if b[0] == 0:
        raise ZeroDivisionError("El primer elemento de la diagonal principal es 0, por lo que no se puede aplicar el algoritmo directamente.")
    C[0] = c[0] / b[0]
    
    for i in range(1, n):
        den[i] = b[i] - (a[i] * C[i-1])
        
        if den[i] == 0:
            raise ZeroDivisionError("Algún denominador es 0, no se puede realizar la división.")
        
        if i != n-1:
            C[i] = c[i] / den[i] 
        
    return den, C
    
    

def st_precomputo( C, den, a , d, dtype = 32):
    
    n = len(d)
    if n != len(den) != len(C)-1:
        raise ValueError("Los vectores no tienen el tamaño adecuado.")
    
    # En la mayoria de las implementaciones el indice de a arranca en 1 en lugar de 0. Para mantener coherencia agregamos un 0.
    a = np.concatenate(([0],a))
    
    if dtype == 32:
        tipo = np.single
    else:
        tipo = np.double
    
    a = np.array(a, tipo)
    C= np.array(C, tipo)
    den = np.array(den, tipo)
    d = np.array(d, tipo)
    
    # Caso i = 1
    d[0] = d[0] / den[0]
    
    
    # Creamos d'
    for i in range( 1, n):
        d[i] = (d[i] - a[i] * d[i-1]) / den[i]
        
    
    # Sustitución
    res = np.zeros(n)
    
    res[n-1] = d[n - 1]
    for i in range(-2, -n - 1, -1):
        res[i] = d[i] - (C[i + 1] * res[i+1])
        
    return res
    
    