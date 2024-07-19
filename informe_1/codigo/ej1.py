import pytest
import numpy as np
import numpy.linalg as lng
from kahan import kahan


def eliminacionGuassianaSinPivoteo ( m , b, dtype =32):
    
    
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
        
    # Primero debemos transformar a nuetra matriz cuadrada en una matriz triangular superior
    
    for i in range(0, n - 1):
        
        if m[i][i] == 0: 
                raise ZeroDivisionError("El algoritmo detecto un 0 en la diagonal. Sin pivoteo no se puede resolver.")
            
        for j in range(i + 1, n):
            
            temp = m[j][i]/m[i][i] 
            
            for k in range(i, n + 1):
                m[j][k] = m[j][k] - (temp * m[i][k])
            
    
            
    # Una vez obtenemos la matriz diagonal superior, toca verificar si el sistema es compatible indeterminado.
    
    if m[n-1][n-1] == 0:
        if m[n-1][n] ==0:
            raise ValueError("El sistema es compatible indeterminado.")
        else:
            raise ValueError("El sistema es incompatible.")
        
    # Resolvemos el sistema.
    
    res = np.zeros(n)
    for i in range(n - 1, -1, -1):
        res[i] = (m[i][n] - kahan(m[i][j] * res[j] for j in range(i + 1, n))) / m[i][i]
    
    res = res.reshape(-1,1)
    return res



"""
Ejemplo donde sin pivoteo no se puede resolver el sistema:

A=
[[ 2,2 ,-1 , 3],
[-2, -2, 0, 0],
[4 ,1, -2 ,4],
[-6 ,-1 ,2 ,-3]] 

b=
[13, -2, 24, -10]

Verificar con: 

eliminacionGuassianaSinPivoteo([[ 2, 2 , -1 , 3],
                                [-2, -2, 0, 0],
                                [ 4, 1, -2 , 4],
                                [-6 ,-1 , 2, -3]], [13, -2, 24, -10])
                                
"""


