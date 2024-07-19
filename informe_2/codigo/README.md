
# Entorno virtual
Para que no haya conflictos con bibliotecas vamos a usar un entorno virtual (para más info ver [este tutorial](https://youtu.be/Y21OR1OPC9A?si=uw5K0eGFtSy4L6vP)). Para instalarlo hay que entrar al directorio del repo y ejecutar:

## Windows
`python -m venv env`

Una vez creado, cada vez que trabajemos en el proyecto hay que activarlo para acceder a todas las dependencias. Para eso ejecutar:

`env\Scripts\activate.bat` # Nota: Si no anda sacar el .bat

## Linux & MacOS
`python3 -m venv env`

Una vez creado, cada vez que trabajemos en el proyecto hay que activarlo para acceder a todas las dependencias. Para eso ejecutar:

`source env/bin/activate`

Para desactivarlo en ambos casos se usa `deactivate`

Con `pip list` chequeamos todas las dependencias que tiene el entorno virtual



## Modalidad
Al pushear código se ignora el entorno virtual. Por lo tanto, cuando se instale una dependencia nueva se tiene que ejecutar el comando `pip freeze > requirements.txt` antes de pushear los cambios. Agregar en el mensaje de commit que se agregó una nueva dependencia. 
Cuando se haga `pull` del nuevo commit basta con ejecutar `pip install -r requirements.txt` para tener todas las dependencias.

## Para datos compartidos con cpp
g++ -shared -fPIC -Llibdl -o eigen_ctypes_test.so eigen_ctypes_test.cpp
Esto crea el archivo `eigen_ctypes_test.so` que es el que puede llamarse desde python.

