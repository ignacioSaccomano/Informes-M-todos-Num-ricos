import pytest 
import numpy as np
import numpy.linalg as lng
from ej1 import eliminacionGuassianaSinPivoteo
from ej2 import eliminacionGuassianaConPivoteoParcial
from ej3_b import eliminacionGaussianaST
from ej3_c import st_precomputo, auxiliar_precomputo

# Para testear nuestros algoritmos estamos utilizando la biblioteca pytest y diversos metodos de numpy.

# Para testear: pytest -s .\tests.py

# Generador de matrices: https://es.planetcalc.com/9081/
# Resolver sistema de ecuaciones lineales con EG: https://matrixcalc.org/es/slu.html

# Todas las implementaciones pueden recibir como ultimo parametro el tipo de dato (single = 32 o double = 64) que se desea utilizar.
# Si no se especifica sera single. En la mayoria de los casos cuando se utiliza double las discrepancias se van a 0.


@pytest.mark.skip() # Si se quiere ejecutar el test especifico borrar o comentar esta linea.
def test_especifico_eliminacionGuassianaSinPivoteo():
    
    tolerancia = 1e-6 # Definimos tolerancia entre vectores respuesta y solucion. Tolerancia de 0.000001
    
    ###########################################################################
    Res1 = eliminacionGuassianaSinPivoteo( 
    [[-19.2 ,88.5, -48.2, -23.0 ,-19.2 ],
    [62.8 ,-98.7 ,53.1, -29.7 ,-57.2 ],
    [-47.8, 3.9, -44.9, 46.8, -67.4 ],
    [0.0 ,-49.9 ,16.2, -63.8 ,-98.8 ],
    [82.7, -95.0 ,32.9, -73.4, -76.0]], [ 70.8, 65.7, -7.7, 28.6, 78.3])
    
    Sol1 = [11592711417751/6578306854597,57325357892958/32891534272985,41534838064896/32891534272985,22415083353411/32891534272985,(-92276339683447)/65783068545970]
    
    assert np.allclose(Res1, Sol1, tolerancia)
    
    ############################################################################
    Res2 = eliminacionGuassianaSinPivoteo(
        [[ 2,1 ,-1 , 3],
         [-2, 0, 0, 0],
         [4 ,1, -2 ,4],
         [-6 ,-1 ,2 ,-3]], [13, -2, 24, -10])
    
    Sol2 = [1, -30, 7, 16] 
       
    assert  np.allclose( Res2, Sol2, tolerancia)
    
    ###################################################################################
    
    # Sistema compatible indeterminado
    
    with pytest.raises(ValueError): eliminacionGuassianaSinPivoteo([[1, 3, 1],
                                                                    [ 0, -16, -5],
                                                                    [ 0, -16, -5]], [2,-11,-15])
    
    #####################################################################################
    
    # Sin pivoteo se rompe
    
    with pytest.raises(ZeroDivisionError): eliminacionGuassianaSinPivoteo([[ 2,2 ,-1 , 3],
                                                                            [-2, -2, 0, 0],
                                                                            [4 ,1, -2 ,4],
                                                                            [-6 ,-1 ,2 ,-3]], [13, -2, 24, -10])
    
    
    
    #######################################################################################
    
    # EJEMPLO DONDE lng.solve() funciona incorrectamente. El siguiente sistema no tiene solucion, pero segun el metodo si.
    
    x = [[ 2, -2 , 2],
        [-2 ,-4,  4],
        [-4 ,-1  ,1]]
    
    y = [ 0,-2, 3]
    
    
    res66 = lng.solve(x, y)

    ######################################################################################
    
    # EJEMPLO DE ERROR NUMERICO         que hacemo?¡?¡
    
    a = [[-3,-2 ,-1],
    [-2 ,-5, -1],
    [-4 , 1 ,-1]]

    b = [-2 ,1 ,-1]

    res = eliminacionGuassianaSinPivoteo(a , b)
    1
    

def test_aleatorio_eliminacionGaussianaSinPivoteo():
    
    tolerancia = 1e-5
    
    cantidad_de_tests = 100
    tamaño_A = 5
    
    exitos = 0
    discrepancias = 0 # Las discrepancias se suelen dar porque tanto .solve() como nuestra implementacion dieron la respuesta incorrecta.
    excepciones_sin_pivoteo = 0  # Variable que lleva registro de la cantidad de sistemas que sin pivoteo no se pudieron resolver.
    excepciones_solve = 0
    excepciones_algo = 0
    
    
    # La funcion lng.solve levanta una excepcion (lng.LinAlgError) si la matriz es singular, es decir, no hay una unica solucion.
    
    for i in range(0,cantidad_de_tests):
        
        # Matrices de enteros. Si se quieren floats usar np.random.uniform(...)
        A = np.random.randint(-5,5,( tamaño_A, tamaño_A))
        b = np.random.randint(-5,5,(tamaño_A,1))
        
        # Transformar el tipo los valores de las matrices de int a float
        
        A = A.astype(np.double)
        b = b.astype(np.double)
        
        
        try:
            
            Sol = lng.solve(A,b)
            Res = eliminacionGuassianaSinPivoteo( A, b, 64)
            
           
            
            if np.allclose(Sol, Res, tolerancia):
                exitos +=1
                
                assert 1
            else:
                # print("\nCantidad de exitos hasta fallar: ", exitos)
                # print("A que genera el error: ", A)
                # print("b que genera elJ error: ", b)
                # print("Res = ", Res)
                # print("Res type = ", type(Res))
                # print("Sol = ", Sol)
                # print("Sol type = ", type(Sol))
                
                discrepancias += 1
                
        except lng.LinAlgError:
            
            excepciones_solve+= 1
        
        except ZeroDivisionError: 
            excepciones_sin_pivoteo += 1
        
        except ValueError:
            excepciones_algo += 1
            
    print("\nTest aleatorio SIN pivoteo:")
    print("\nCantidad de exitos: ", exitos)
    print("Cantidad de excepciones de nuestro algo: ", excepciones_algo)
    print("Cantidad de excepciones de .solve() = ", excepciones_solve)
    print("Cantidad de excepciones por falta de pivoteo: "  , excepciones_sin_pivoteo)
    print("Cantidad de discrepancias: ", discrepancias)
    
    
#################################################################################################

def test_aleatorio_eliminacionGuassianaConPivoteoParcial():

    tolerancia = 1e-6

    cantidad_de_tests = 100
    tamaño_A = 5

    exitos = 0
    discrepancias = 0
    excepciones_solve = 0 # Estas excepciones dependen de quien se ejecuta primero, si el .solve o nuestra implementacion. Esto porque si se produce una excepcion en el primero no se ejecuta el segundo.
    excepciones_algo = 0

    # La funcion lng.solve() levanta una excepcion (lng.LinAlgError) si la matriz es singular, es decir, no hay una unica solucion.
    # A su vez, .solve() devuelve un ndarray de singles o doubles, depende de la matriz entrante.
    
    for i in range(0,cantidad_de_tests):
        
        # Matrices de enteros. Si se quieren floats usar np.random.uniform(...)
        A = np.random.randint(-5,5,( tamaño_A, tamaño_A))
        b = np.random.randint(-5,5,(tamaño_A,1))
        
        # Transformar el tipo los valores de las matrices de int a float
        
        A = A.astype(np.double)     #Por que son necesarias estas 2 lineas???? raro
        b = b.astype(np.double)
        
        
        try:
            
            Sol = lng.solve(A,b)
            
            Res = eliminacionGuassianaConPivoteoParcial( A, b, tolerancia, 64)
            
            
            if np.allclose(Sol, Res, tolerancia):
                exitos +=1
                assert 1
            else:
                
                # print("\nCantidad de exitos hasta fallar: ", exitos)
                # print("A que genera el error: ", A)
                # print("b que genera el error: ", b)
                # print("Res = ", Res)
                # print("Res type = ", type(Res))
                # print("Sol = ", Sol)
                # print("Sol type = ", type(Sol))
                
                discrepancias +=1
                
                
        except lng.LinAlgError:
            excepciones_solve += 1
            
        except ValueError:
            excepciones_algo += 1
        
        
    print("\nTest aleatorio CON pivoteo:")
    print("\nCantidad de exitos: ", exitos)
    print("Cantidad de excepciones de nuestro algo: ", excepciones_algo)
    print("Cantidad de excepciones de .solve() = ", excepciones_solve)
    print("Cantidad de discrepancias: ", discrepancias)

############################################################################################

def test_aleatorio_eliminacionGuassianaST():

    tolerancia = 1e-6

    cantidad_de_tests = 100
    tamaño = 5

    exitos = 0
    discrepancias = 0
    excepciones_solve = 0 # Estas excepciones dependen de quien se ejecuta primero, si el .solve o nuestra implementacion. Esto porque si se produce una excepcion en el primero no se ejecuta el segundo.
    excepciones_algo = 0
    excepciones_algo_ZeroDivision = 0

    # La funcion lng.solve() levanta una excepcion (lng.LinAlgError) si la matriz es singular, es decir, no hay una unica solucion.
    # A su vez, .solve() devuelve un ndarray de singles o doubles, depende de la matriz entrante.
    
    for i in range(0,cantidad_de_tests):
        
        # Vectores de enteros. Si se quieren floats usar np.random.uniform(...)
        a = np.random.randint(-5,5,(tamaño-1))
        b = np.random.randint(-5,5,(tamaño))
        c = np.random.randint(-5,5,(tamaño-1))
        d = np.random.randint(-5,5,(tamaño))
        
        # Creamos la matriz tridiagonal para darsela a .solve()
        
        Tri_A = np.diag(b) + np.diag(c, 1) + np.diag(a, -1)
        Tri_A = Tri_A.astype(np.double)
        Tri_B = d.astype(np.double)
        
        try:
            
            Sol = lng.solve( Tri_A, Tri_B)
            
            Res = eliminacionGaussianaST(a,b,c,d, 64)
            
            
            if np.allclose(Sol, Res, tolerancia):
                exitos +=1
                assert 1
            else:
                
                # print("\nCantidad de exitos hasta fallar: ", exitos)
                # print("A que genera el error: ", A)
                # print("b que genera el error: ", b)
                # print("Res = ", Res)
                # print("Res type = ", type(Res))
                # print("Sol = ", Sol)
                # print("Sol type = ", type(Sol))
                
                discrepancias +=1
                
                
        except lng.LinAlgError:
            excepciones_solve += 1
            
        except ValueError:
            excepciones_algo += 1
            
        except ZeroDivisionError:
            excepciones_algo_ZeroDivision += 1
        
        
    print("\nTest aleatorio ST:")
    print("\nCantidad de exitos: ", exitos)
    print("Cantidad de excepciones de nuestro algo: ", excepciones_algo)
    print("Cantidad de excepciones de .solve() = ", excepciones_solve)
    print("Cantidad de ZeroDivisions: ", excepciones_algo_ZeroDivision)
    print("Cantidad de discrepancias: ", discrepancias)

############################################################################################

def test_aleatorio_ST_precomputo():

    tolerancia = 1e-6

    cantidad_de_tests = 10
    cantidad_de_ds = 10
    tamaño = 5

    exitos = 0
    discrepancias = 0
    excepciones_solve = 0 # Estas excepciones dependen de quien se ejecuta primero, si el .solve o nuestra implementacion. Esto porque si se produce una excepcion en el primero no se ejecuta el segundo.
    excepciones_algo = 0
    excepciones_algo_ZeroDivision = 0

    # La funcion lng.solve() levanta una excepcion (lng.LinAlgError) si la matriz es singular, es decir, no hay una unica solucion.
    # A su vez, .solve() devuelve un ndarray de singles o doubles, depende de la matriz entrante.
    
    for i in range(0,cantidad_de_tests):
        
        # Vectores de enteros. Si se quieren floats usar np.random.uniform(...)
        a = np.random.randint(-5,5,(tamaño-1))
        b = np.random.randint(-5,5,(tamaño))
        c = np.random.randint(-5,5,(tamaño-1))
        
        
        
        # Creamos la matriz tridiagonal para darsela a .solve()
        Tri_A = np.diag(b) + np.diag(c, 1) + np.diag(a, -1)
        Tri_A = Tri_A.astype(np.double)
        
        
        try:
            den,C = auxiliar_precomputo(a,b,c, 64)
            
            
            for j in range(0 , cantidad_de_ds):
                
                d = np.random.randint(-5,5,(tamaño))
                Tri_b = d.astype(np.double)
                
                Res = st_precomputo(C,den, a, Tri_b, 64)
                
                Sol = lng.solve( Tri_A, Tri_b)
                
                
                if np.allclose(Sol, Res, tolerancia):
                    exitos +=1
                    assert 1
                else:
                    
                    # print("\nCantidad de exitos hasta fallar: ", exitos)
                    # print("A que genera el error: ", A)
                    # print("b que genera el error: ", b)
                    # print("Res = ", Res)
                    # print("Res type = ", type(Res))
                    # print("Sol = ", Sol)
                    # print("Sol type = ", type(Sol))
                    
                    discrepancias +=1
            
            
            
                
                
        except lng.LinAlgError:
            excepciones_solve += 1
            
        except ValueError:
            excepciones_algo += 1
            
        except ZeroDivisionError:
            excepciones_algo_ZeroDivision += 10
        
        
    print("\nTest aleatorio ST_precomputo:")
    print("\nCantidad de exitos: ", exitos)
    print("Cantidad de excepciones de nuestro algo: ", excepciones_algo)
    print("Cantidad de excepciones de .solve() = ", excepciones_solve)
    print("Cantidad de ZeroDivisions: ", excepciones_algo_ZeroDivision)
    print("Cantidad de discrepancias: ", discrepancias)

    
############################################################################################

#test_aleatorio_ST_precomputo()


#test_aleatorio_eliminacionGuassianaST()
#test_aleatorio_eliminacionGaussianaSinPivoteo()
1