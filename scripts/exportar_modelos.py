import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Paths
CACHE = os.path.join("outputs", "embeddings_v2.npz")
MODEL_DIR = os.path.join("outputs", "modelos")
os.makedirs(MODEL_DIR, exist_ok=True)

def exportar_todos():
    if not os.path.exists(CACHE):
        print(f"Error: No se encontró el archivo de embeddings cached en {CACHE}.")
        return
        
    c = np.load(CACHE, allow_pickle=True)
    X = c["X"]
    y = c["y"] # Contains string labels 'Enojo' and 'Tristeza'
    
    print(f"Entrenando y exportando modelos usando {len(y)} muestras...")
    
    # Configure models with probability support
    modelos = {
        "knn_k3": Pipeline([
            ("s", StandardScaler()), 
            ("c", KNeighborsClassifier(n_neighbors=3))
        ]),
        "knn_k5": Pipeline([
            ("s", StandardScaler()), 
            ("c", KNeighborsClassifier(n_neighbors=5))
        ]),
        "svm_lineal": Pipeline([
            ("s", StandardScaler()), 
            ("c", SVC(kernel="linear", C=1, probability=True, random_state=42))
        ]),
        "svm_rbf": Pipeline([
            ("s", StandardScaler()), 
            ("c", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42))
        ]),
        "logreg": Pipeline([
            ("s", StandardScaler()), 
            ("c", LogisticRegression(max_iter=2000, C=1, random_state=42))
        ]),
        "rf": Pipeline([
            ("s", StandardScaler()), 
            ("c", RandomForestClassifier(n_estimators=200, random_state=42))
        ])
    }
    
    for name, pipe in modelos.items():
        pipe.fit(X, y)
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(pipe, path)
        print(f"  Modelo '{name}' guardado en {path}")

if __name__ == "__main__":
    exportar_todos()
