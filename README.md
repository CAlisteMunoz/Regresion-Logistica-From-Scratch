# Proyecto de Regresión Logística

Este proyecto implementa un modelo de regresión logística en Python, utilizando un conjunto de datos cargado desde un archivo CSV. El proyecto incluye el preprocesamiento de datos, entrenamiento del modelo y evaluación de su rendimiento.

## Estructura del Proyecto

```
regresion-logistica-proyecto
├── src
│   ├── __init__.py          # Inicialización del paquete
│   ├── ejemplo.py           # Ejemplo de uso del modelo
│   ├── regresion_logistica.py # Implementación del modelo de regresión logística
│   └── preprocesamiento.py  # Carga y preprocesamiento de datos
├── data
│   └── dataset.csv          # Conjunto de datos en formato CSV
├── tests
│   └── test_regresion.py    # Pruebas unitarias
├── requirements.txt          # Dependencias del proyecto
├── .gitignore                # Archivos y directorios a ignorar por Git
└── README.md                 # Documentación del proyecto
```

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