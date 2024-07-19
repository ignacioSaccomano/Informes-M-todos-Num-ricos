import numpy as np
from interfaz import obtenerDatos

def pca(X, p):  # Y es matriz de datos de dev, X de entrenamiento

    Cx = (X.T @ X) / (X.shape[0] - 1)   # 2. Armamos la matriz de covarianza
    
    
    # Aca para diagonalizar llamamos al metodo de la potencia con deflacion. Lo hacemos para la cantidad maxima de componentes principales que queremos probar. Un numero razonable es 300
    autovalores, V, _ = obtenerDatos(Cx, p)
    # Debido a que el metodo de la potencia encuentra los autovectores asociados a lambdas mas grandes, las columnas estan ordenadas de mayor a menor relevancia en terminos de varianza

    return autovalores, V
