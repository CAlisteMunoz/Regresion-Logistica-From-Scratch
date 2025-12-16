from pathlib import Path
import argparse
import numpy as np
from src.regresion_logistica import reg_log
from src.preprocesamiento import cargar_y_preprocesar_datos

def main():
    parser = argparse.ArgumentParser(description="Entrenar regresión logística desde CSV")
    parser.add_argument("--csv", "-c", default="data/dataset.csv", help="Ruta al CSV")
    parser.add_argument("--target", "-t", default=None, help="Nombre de la columna objetivo")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=200000)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--armijo", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    res = cargar_y_preprocesar_datos(csv_path, target=args.target, test_size=args.test_size,
                                     random_state=42, return_metadata=True)

    if isinstance(res, tuple):
        if len(res) == 5:
            X_train, X_test, y_train, y_test, metadata = res
        else:
            raise ValueError("Formato inesperado devuelto por cargar_y_preprocesar_datos")
    else:
        raise ValueError("cargar_y_preprocesar_datos no devolvió una tupla")

    print("[ejemplo] X_train.shape, X_test.shape:", X_train.shape, X_test.shape)
    print("[ejemplo] y_train distrib:", np.bincount(y_train), " y_test distrib:", np.bincount(y_test))
    print("[ejemplo] metadata resumen:", {k: (type(v), (len(v) if hasattr(v,'__len__') else None)) for k,v in metadata.items()})

    modelo = reg_log(epsilon=args.epsilon, max_iter=args.max_iter, eta=args.eta, m=10, verbose=True)
    modelo.entrenar(X_train, y_train, armijo=args.armijo)

    print("Pesos:", modelo.w.ravel())
    print("Costo final:", modelo.costo_final)
    print("Accuracy:", np.mean(modelo.predict(X_test) == y_test))

if __name__ == "__main__":
    main()