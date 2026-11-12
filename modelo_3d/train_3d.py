# modelo_3d/train_3d.py
import os
import sys
import json
import numpy as np

# Ajustar la ruta para importar los módulos de la carpeta raíz
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)

from FeatureExtractor import Extractor3D
from Trainer import build_sequences, train_classifier
from ModelRepository import ModelRepository

if __name__ == "__main__":
    print("1. Cargando dataset 3D...")
    data_path = os.path.join(os.path.dirname(__file__), "glifo-3d.json")
    
    if not os.path.exists(data_path):
        print(f"ERROR: No se encontró {data_path}")
        sys.exit(1)
        
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Instanciar tu nuevo Extractor3D
    extractor = Extractor3D(k_dir=16, min_angle_dist=0.05)

    print("2. Construyendo secuencias...")
    sequences = build_sequences(raw_data, extractor, min_len=2)

    print("3. Entrenando HMMs (esto puede tardar unos segundos)...")
    # Entrenamos con 6 estados ocultos y 30 iteraciones
    classifier = train_classifier(sequences, n_states=6, n_symbols=extractor.n_symbols, n_iter=30)

    out_path = os.path.join(os.path.dirname(__file__), "models_3d.json")
    print("4. Guardando modelos...")
    ModelRepository(out_path).save(classifier)
    print(f"¡Éxito! Modelo 3D guardado en: {out_path}")