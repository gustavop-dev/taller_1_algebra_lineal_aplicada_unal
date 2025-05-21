# Taller 1 - Álgebra Lineal Aplicada UNAL

Este repositorio contiene la implementación de varios algoritmos relacionados con aplicaciones del álgebra lineal.

## Estructura del Repositorio

El repositorio está organizado en dos carpetas principales:

### 1. Escaleras y Toboganes

Implementación del cálculo de la esperanza del número de lanzamientos necesarios para terminar un juego de Escaleras y Toboganes:

- **primer_punto.py**: Implementación base del cálculo utilizando un dado justo.
- **segundo_punto.py**: Versión mejorada que permite utilizar un dado sesgado con probabilidades personalizables.
- **cuarto_punto.py**: Algoritmo para encontrar la distribución óptima de probabilidades en el dado que minimiza el número esperado de tiradas.

### 2. Conmutatividad

Implementación de algoritmos para encontrar matrices que conmutan con una matriz dada:

- **punto_inicial.py**: Implementación general para calcular una base ortonormal del espacio de matrices que conmutan con una matriz arbitraria.
- **matriz_triangular.py**: Versión optimizada para matrices triangulares que evita la costosa descomposición SVD.

## Requisitos

- Python 3.7 o superior
- NumPy (≥ 1.20.0)
- SciPy (≥ 1.7.0)

## Instalación

### Crear un entorno virtual

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Instalar las dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Escaleras y Toboganes

Para ejecutar el cálculo con un tablero personalizado:

```bash
# Primer punto (dado justo)
python escaleras_y_toboganes/primer_punto.py tablero.json

# Segundo punto (dado sesgado)
python escaleras_y_toboganes/segundo_punto.py tablero.json 0.1,0.1,0.2,0.2,0.2,0.2

# Cuarto punto (encontrar distribución óptima)
python escaleras_y_toboganes/cuarto_punto.py
```

El archivo JSON del tablero debe tener el siguiente formato:
```json
{
  "length": 9,
  "links": [[2, 4], [7, 3], [6, 8]]
}
```

Donde `length` es el tamaño del tablero y `links` son los enlaces (escaleras y toboganes) representados como pares [inicio, fin].

### Conmutatividad

```bash
# Algoritmo general
python conmutatividad/punto_inicial.py

# Versión optimizada para matrices triangulares
python conmutatividad/matriz_triangular.py
```

Los scripts incluyen ejemplos de demostración que se ejecutan automáticamente. 