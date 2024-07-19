import numpy as np
from knn import crossValidationPca, calcular_exactitud
import matplotlib.pyplot as plt
from pca import pca

X_train = np.loadtxt("datos/X_train.csv", delimiter=",")
y_train = np.loadtxt("datos/y_train.csv", delimiter=",").astype(int)
X_test = np.loadtxt("datos/X_test.csv", delimiter=",")
y_test = np.loadtxt("datos/y_test.csv", delimiter=",").astype(int)


X_train = X_train - X_train.mean(axis=0)
X_test = X_test - X_test.mean(axis=0)

# Primero calculamos la matriz de autovectores V para el cambio de base.

# _, V_test = pca(X_train, 784)    # Primero para el caso final
# np.savetxt('resultados/autovectores_final.txt', V_test)
V_test = np.loadtxt("resultados/autovectores_final.txt")


MAX_P = 784 # Exploramos solo hasta p = 100 porque acumula un 90% de varianza

V_i = np.empty((5, 784,MAX_P))  # 5 matrices de autovectores, uno para cada fold. Usaremos solo 100 componentes porque acumulan un 90% de varianza

# inicio = 0
# final = 1000

for j in range(5):
    # X_newtrain = np.concatenate((X_train[:inicio, :], X_train[final:,:]), axis=0)
    # _, V_i[j] = pca(X_newtrain, 784)   # Precalculamos la matriz correspondiente a cada fold
    # np.savetxt(f'resultados/autovectores{j}.txt', V_i[j])
    # inicio += 1000
    # final += 1000
    V_i[j] = np.loadtxt(f"resultados/autovectores{j}.txt")


# Vamos a hacer 2 loops. Una para k y otro para p <= 100 (El resto suma poco)
rangoP = np.concatenate([np.linspace(1,15, 15, dtype=int), np.array([100, 200, 300, 400, 500, 600, 700, 784], dtype=int)])
cantK = 30
resultados = np.empty(shape=(cantK,len(rangoP)))
def mejor_par():
    for k in range(1, cantK + 1):
        for i, p in enumerate(rangoP): 
            resultados[k-1][i] = crossValidationPca(X_train, y_train, k, V_i[:,:,:p])

    max_k, max_p = np.unravel_index(np.argmax(resultados), shape=resultados.shape)  # argmax devuelve el indice como si la matriz fuera plana, con unravelindex reconstruimos los indices

    porcentaje = resultados[max_k][max_p]
    np.savetxt("resultados/resultados_3d", resultados)
    return (max_k + 1), (rangoP[max_p]), porcentaje

x = mejor_par()
# np.savetxt('optimo',x)
# resultados = np.loadtxt("resultados/resultados_3d")
# x = np.loadtxt('optimo').astype(int)
plt.rcParams["xtick.labelsize"] = 6
plt.pcolor(resultados)
plt.xticks(np.arange(len(rangoP)), rangoP, rotation=45)
plt.yticks(np.arange(cantK),np.arange(1, cantK + 1))
cbar = plt.colorbar()
cbar.set_label("Efectividad")
plt.ylabel('#vecinos')
plt.xlabel("#componentes")
plt.title("Exploración de hiperparámetros con PCA")
plt.savefig("graficos/mapa_de_calor.png")

# Finalmente, calculamos para el mejor k y p la prueba con los datos de testing
V_test_split = V_test[:,:x[1]]
X_train = X_train @ V_test_split
X_test = X_test @ V_test_split
final = calcular_exactitud(X_train, y_train, X_test, y_test, x[0])

with open('final.txt', 'w') as file:
    file.write("Resultado final = " + repr(final) + '\n')

    
