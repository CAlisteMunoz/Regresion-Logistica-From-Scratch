import unittest
import pandas as pd
import numpy as np
from src.regresion_logistica import reg_log
from src.preprocesamiento import cargar_y_preprocesar_datos

class TestRegresionLogistica(unittest.TestCase):

    def setUp(self):
        # Cargar y preprocesar los datos
        self.X_train, self.X_test, self.y_train, self.y_test = cargar_y_preprocesar_datos('data/dataset.csv')

    def test_modelo_entrenamiento(self):
        modelo = reg_log(epsilon=1e-3, max_iter=200_000, eta=.001)
        modelo.entrenar(self.X_train, self.y_train)
        
        # Verificar que los pesos no sean nulos
        self.assertIsNotNone(modelo.w)
        self.assertGreater(np.linalg.norm(modelo.w), 0)

    def test_accuracy(self):
        modelo = reg_log(epsilon=1e-3, max_iter=200_000, eta=.001)
        modelo.entrenar(self.X_train, self.y_train)
        
        accuracy = np.mean(modelo.predict(self.X_test) == self.y_test)
        # Verificar que la precisión sea mayor que un umbral
        self.assertGreater(accuracy, 0.5)

if __name__ == '__main__':
    unittest.main()