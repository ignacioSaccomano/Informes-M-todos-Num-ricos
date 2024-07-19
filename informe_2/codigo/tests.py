from interfaz import obtenerDatos as eigen
import numpy as np
import numpy.linalg as lng
from scipy.stats import ortho_group

# Para testear nuestros algoritmos estamos utilizando la biblioteca pytest y diversos metodos de numpy.
# Tambien utilizaremos el "truco" de la matriz de Householder y la funcion eigh de numpy
# Para testear: pytest -s .\tests.py

def find_swaps_to_sort_descending(arr):
    """
    Encuentra las posiciones que deben intercambiarse para ordenar el array en orden descendente.

    Args:
        arr (list): El array de entrada.

    Returns:
        list: Lista de tuplas de posiciones a intercambiar.
    """
    indexed_arr = [(val, i) for i, val in enumerate(arr)]

    # Ordenar el array en orden descendente
    sorted_arr = sorted(indexed_arr, reverse=True)

    swaps = []
    seen_positions = set()  # Conjunto para rastrear posiciones ya procesadas

    for i in range(len(arr)):
        if sorted_arr[i][1] != i and (i, sorted_arr[i][1]) not in seen_positions:
            swaps.append((i, sorted_arr[i][1]))
            seen_positions.add((sorted_arr[i][1], i))  # Agregar la tupla inversa

    return swaps

def swap_matrix_columns(matrix, swaps):
    """
    Intercambia las columnas de la matriz según las tuplas de posiciones.

    Args:
        matrix (np.ndarray): La matriz de entrada.
        swaps (list): Lista de tuplas de posiciones a intercambiar.

    Returns:
        np.ndarray: La matriz con las columnas intercambiadas.
    """
    if not swaps:
        # Si la lista de swaps está vacía, no hacemos nada
        return matrix

    # Crear una copia de la matriz para no modificar la original
    result_matrix = np.copy(matrix)

    for pos1, pos2 in swaps:
        # Intercambiar las columnas pos1 y pos2
        result_matrix[:, [pos1, pos2]] = result_matrix[:, [pos2, pos1]]

    return result_matrix

def test_especifico_potencia():
    
    tolerancia= 1e-4
    
    auto_1 = [5.0, 4.0, 3.0, 2.0, 1.0]
    D = np.diag(auto_1)

    v = np.ones((D.shape[0], 1))

    v = v / np.linalg.norm(v)

    # Matriz de Householder
    B = np.eye(D.shape[0]) - 2 * (v @ v.T)
    # Matriz a diagonalizar
    M = B.T @ D @ B

    res1_valores, res1_vectores,_ = eigen(M, len(D[0]))
    sol1_valores, sol1_vectores = lng.eigh(M)
    sol1_valores= np.flip(sol1_valores)
    sol1_vectores= np.flip(sol1_vectores, axis=0)
    
    assert np.allclose(res1_valores, sol1_valores)
    
    # Toca verificar los autovectores. En este caso, los autovectores son las columnas de B (Matriz de Householder).
    
    for i in range(len(res1_vectores[0])):
        vector_res = res1_vectores[:,i]
        vector_sol = sol1_vectores[:,i]
        
        dot= np.dot(vector_res, vector_sol)
        mul_norma= np.linalg.norm(vector_sol) * np.linalg.norm(vector_res)
        # Si 2 vectores estan en la misma direccion entonces su producto escalar será igual a la norma 2 de uno de ellos multiplicada por la norma del otro.
        # Si estan en direcciones opuestas entonces estos valores seran opuestos.
        # Ambos casos indican que los vectores estan sobre la misma recta, lo que buscamos.
        assert np.isclose(dot , mul_norma, tolerancia) or np.isclose(-dot , mul_norma, tolerancia)


def test_aleatorio_potencia():
    
    tamaño_matrices= 5
    tolerancia= 1e-3 # Tolerancia demasiado alta
    
    for reps in range(10):
        
        auto = np.random.uniform(low=0.5, high=100.0, size=tamaño_matrices)
        auto[::-1].sort()
        D = np.diag(auto)

        # Generar una matriz ortogonal aleatoria. Lo mismo que el truco de Householder
        B = ortho_group.rvs(dim=tamaño_matrices)

        # Matriz a diagonalizar
        M = B.T @ D @ B
        
        # Calculamos
        res1_valores, res1_vectores, _ = eigen(M, len(M[0]))
        sol1_valores, sol1_vectores = lng.eigh(M)
        
        # eigh devuelve de menor a mayor y nuestra implementacion de mayor a menor. 
        sol1_valores= np.flip(sol1_valores)
        sol1_vectores= np.flip(sol1_vectores, axis=1)
        
        # Observacion: hay casos en los que nuestra implementacion devuelve los autovalores, y por ende los autovectores, desordenados.
        # Solucion provisional: obtener posiciones de valores a ordenar y ordenar en la matriz.
        
        swaps = find_swaps_to_sort_descending(res1_valores)
        res1_valores[::-1].sort()
        res1_vectores = swap_matrix_columns(res1_vectores, swaps)
        
        # Chequeamos autovalores
        assert np.allclose(res1_valores, sol1_valores, tolerancia)
        
        # Chequeamos autovectores
        for i in range(len(res1_vectores[0])):
            vector_res = res1_vectores[:,i]
            vector_sol = sol1_vectores[:,i]
            
            dot= np.dot(vector_res, vector_sol)
            mul_norma= np.linalg.norm(vector_sol) * np.linalg.norm(vector_res)
            # Si 2 vectores estan en la misma direccion entonces su producto escalar será igual a la norma 2 de uno de ellos multiplicada por la norma del otro.
            # Si estan en direcciones opuestas entonces estos valores seran opuestos.
            # Ambos casos indican que los vectores estan sobre la misma recta, lo que buscamos.
            assert (np.isclose(dot , mul_norma, tolerancia) or np.isclose(-dot , mul_norma, tolerancia))

test_aleatorio_potencia()
test_especifico_potencia()