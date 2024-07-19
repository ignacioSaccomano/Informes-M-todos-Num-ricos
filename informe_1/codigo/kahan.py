from numba import jit
'''
    Metodo para sumatorias que disminuye el error (sacado del notebook del primer labo)
'''
    # Por algo el decorator me da error asi que lo saco
def kahan(lista):
    # Accumulator
    suma = 0.0

    #A running compensation for lost low-order bits.
    c = 0.0

    for x in lista:
        y = x - c
        t = suma + y
        c = (t - suma) - y #recupero los digitos menos significativos de y para la próxima
        suma = t
    return suma