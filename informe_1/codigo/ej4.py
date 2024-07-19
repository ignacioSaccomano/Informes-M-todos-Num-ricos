import matplotlib.pyplot as plt
import numpy as np
from ej3_b import eliminacionGaussianaST

n = 101

# Creamos los d
d_a = np.zeros(n)
temp = (n//2) + 1
d_a[ temp ] = 4 / n

d_b = np.full( n, 4/(n**2), np.double)

d_c = np.zeros(n)
for i in range(0,n):
    d_c[i] = (-1 + 2*i/(n-1)) * (12/n**2)

# Creamos la matriz tridiagonal del operador laplaciano.

a = np.full( n -1 , 1, np.double)
b = np.full( n  , -2, np.double)
c = np.full( n -1 , 1, np.double)


res_a = eliminacionGaussianaST( a, b, c, d_a, 64)
res_b = eliminacionGaussianaST( a, b, c, d_b, 64)
res_c = eliminacionGaussianaST( a, b, c, d_c, 64)


x = np.arange(0,101)
plt.plot( x,res_a, "-",label='(a)')
plt.plot( x,res_b, "-",label='(b)')
plt.plot( x,res_c, "-",label='(c)')
plt.xlabel('x')
plt.ylabel('u')
plt.legend(loc='lower left')
plt.show()