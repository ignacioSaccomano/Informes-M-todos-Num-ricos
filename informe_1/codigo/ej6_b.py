import matplotlib.pyplot as plt
import numpy as np
from ej3_b import eliminacionGaussianaST


n = 101
r = 10
m = 1000
alpha = 1
lista = []

u = np.zeros(n)

desde = (n//2) - r

hasta = (n//2) + r

for i in range(desde, hasta):
    u[i]= 1


a = np.full(n - 1, alpha)
b = np.full(n , 1- (2*alpha))
c = a 

A = np.diag(b) + np.diag(c, 1) + np.diag(a, -1)
A= A.astype(np.double)
lista.append(u)


for k in range(0,m):
    u = np.matmul(A, u)
    lista.append(u)

lista = np.array(lista)
lista = lista.T

plt.pcolor(lista, cmap = 'hot')
plt.xlabel('k')
plt.ylabel('x')
plt.title('Simulacion de difusión para α = 1')
plt.colorbar()
plt.show()