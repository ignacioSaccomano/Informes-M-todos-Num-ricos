import numpy as np
import os

def limpiar():
      os.system("echo > iteraciones.txt")  
      os.system("echo > autovalores.txt")  
      os.system("echo > input_data.txt")  
      os.system("echo > autovectores.txt")  

def obtenerDatos(matriz, niter):
    limpiar()
    matriz = matriz.astype(np.float64)
    with open('input_data.txt','a') as f:
            f.write(f"{matriz.shape[0]} {matriz.shape[1]}\n")
            np.savetxt(f,matriz, newline="\n")
            f.write(f"{niter}\n")

    os.system("./metodoPotencia input_data.txt autovalores.txt autovectores.txt iteraciones.txt")
    autovalores = np.loadtxt("autovalores.txt")
    autovectores = np.loadtxt("autovectores.txt")
    iteraciones = np.loadtxt("iteraciones.txt")
    return autovalores, autovectores, iteraciones