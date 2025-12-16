# Implementación de Regresión Logística (From Scratch)

**Curso:** Programación 2  
**Institución:** Escuela de Ingeniería, Universidad de O'Higgins  
**Autores:** Pedro Alemany, Camilo Aliste

## Descripción del Proyecto

Este repositorio contiene una implementación completa y manual de un clasificador binario basado en **Regresión Logística**, desarrollado sin la dependencia de librerías de alto nivel para el entrenamiento de modelos (como scikit-learn). 

El objetivo principal de este proyecto es demostrar la comprensión profunda de los fundamentos matemáticos detrás del aprendizaje supervisado, abordando desde cero:
1. La formulación probabilística del modelo.
2. El diseño de la función de costo (Log-Loss).
3. La implementación de algoritmos de optimización numérica avanzada (L-BFGS).
4. El manejo de estabilidad numérica y convergencia.

## Fundamentos Teóricos

El modelo se basa en la modelación de la probabilidad condicional $P(Y=1|X)$ mediante la función sigmoide. La optimización de los parámetros se realiza minimizando la entropía cruzada.

### 1. Modelo Matemático
La predicción de probabilidad se define como:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Donde $z = w^T x + b$.

### 2. Optimización Numérica (L-BFGS)
A diferencia de implementaciones básicas que utilizan Descenso de Gradiente estocástico o por lotes, esta implementación utiliza el algoritmo **L-BFGS (Limited-memory BFGS)**. Este es un método Quasi-Newton que aproxima la inversa de la matriz Hessiana utilizando la recursión de dos bucles (*two-loop recursion*), lo que permite una convergencia más rápida y estable en problemas convexos.

### 3. Control de Convergencia
Para garantizar la estabilidad del descenso, se implementa una búsqueda de línea (*Line Search*) con **Condición de Armijo**, asegurando que cada paso de actualización reduzca efectivamente la función de costo global.

## Estructura del Repositorio

La arquitectura del proyecto sigue un diseño modular para separar la lógica de negocio, el procesamiento de datos y la ejecución.

```text
Regresion-Logistica-From-Scratch/
├── data/
│   └── dataset.csv             # Conjunto de datos crudo
├── src/
│   ├── __init__.py
│   ├── regresion_logistica.py  # Clase reg_log con implementación de L-BFGS
│   ├── preprocesamiento.py     # Rutinas de limpieza, normalización y escalado
│   └── ejemplo.py              # Script de orquestación principal
├── tests/
│   └── test_regresion.py       # Pruebas unitarias para validar integridad numérica
├── requirements.txt            # Dependencias del entorno Python
└── README.md                   # Documentación técnica del proyecto

## Instalación

1. Clona el repositorio:
   ```
   git clone <URL_DEL_REPOSITORIO>
   cd regresion-logistica-proyecto
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Asegúrate de que el archivo `dataset.csv` esté en la carpeta `data`.
2. Ejecuta el archivo `ejemplo.py` para ver un ejemplo de uso del modelo de regresión logística:
   ```
   python src/ejemplo.py
   ```

## Pruebas

Para ejecutar las pruebas unitarias, utiliza el siguiente comando:
```
python -m unittest discover -s tests
```

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT.
