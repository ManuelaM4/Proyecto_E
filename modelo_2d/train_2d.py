"""
train_2d.py
Script de entrenamiento offline para el modo 2D.
"""
import json
import os
import sys
from pathlib import Path

# Asegurar que encontramos los módulos de la raíz
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from FeatureExtractor import Extractor2D
from Trainer import build_sequences, split_dataset, train_classifier, evaluate
from ModelRepository import ModelRepository

def main():
    print("Iniciando entrenamiento 2D...")
    
    extractor = Extractor2D(k_dir=16, min_dist=15.0) 
    dataset_path = Path(__file__).parent / 'glifo-dataset-2026-11-12.json'
    
    if not dataset_path.exists():
        print(f" ERROR: No se encontró el dataset en {dataset_path}")
        return

    print(f" Dataset encontrado: {dataset_path.name}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print("Extrayendo características de los trazos...")
    data = build_sequences(raw_data, extractor, min_len=3)
    
    train_data, test_data = split_dataset(data, train_ratio=0.8)
    
    print("Entrenando modelos HMM (Baum-Welch)...")
    classifier = train_classifier(train_data, n_states=6, n_symbols=extractor.n_symbols, n_iter=30)
    
    print("\nEvaluando modelo...")
    accuracy, por_letra = evaluate(classifier, test_data)
    print(f"Precisión global: {accuracy * 100:.2f}%")
    
    output_path = Path(__file__).parent / 'models_2d.json'
    repo = ModelRepository(output_path)
    repo.save(classifier)
    print(f"\n Modelos guardados exitosamente en: {output_path}")

if __name__ == "__main__":
    main()