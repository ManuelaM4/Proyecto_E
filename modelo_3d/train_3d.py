# modelo_3d/train_3d.py
import os
import sys
import json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)

from FeatureExtractor import Extractor3D
from Trainer import train_classifier, evaluate
from ModelRepository import ModelRepository

MIN_LEN = 3
TRAIN_RATIO = 0.8
SEED = 42

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "glifo-3d.json")
    aug_path = os.path.join(here, "glifo-3d-aug.json")

    if not os.path.exists(data_path):
        print(f"ERROR: No se encontró {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    aug = {}
    if os.path.exists(aug_path):
        with open(aug_path, "r", encoding="utf-8") as f:
            aug = json.load(f)
        print("Aumentos encontrados.")

    extractor = Extractor3D(k_dir=16, min_angle_dist=0.03)
    rng = np.random.default_rng(SEED)
    train, test = {}, {}

    print("Extrayendo características...")
    for ch, muestras in raw.items():
        aug_letra = aug.get(ch, [])
        idx = rng.permutation(len(muestras))
        corte = int(TRAIN_RATIO * len(muestras))
        train[ch], test[ch] = [], []
        for pos, i in enumerate(idx):
            seq = extractor.extract(muestras[i])
            if len(seq) < MIN_LEN:
                continue
            if pos < corte:
                train[ch].append(seq)
                copias = aug_letra[i] if i < len(aug_letra) else []
                for copia in copias:
                    s = extractor.extract(copia)
                    if len(s) >= MIN_LEN:
                        train[ch].append(s)
            else:
                test[ch].append(seq)

    print("Entrenando HMMs (Baum-Welch)...")
    classifier = train_classifier(train, n_states=6,
                                  n_symbols=extractor.n_symbols, n_iter=30)

    print("\nEvaluando...")
    acc, por_letra = evaluate(classifier, test)
    print(f"Precisión global: {acc * 100:.2f}%")
    peores = sorted(por_letra.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))[:8]
    print("Letras más difíciles:", [(c, f"{a}/{t}") for c, (a, t) in peores])

    # Confusiones más frecuentes (real -> predicha)
    conf = {}
    for ch, seqs in test.items():
        for s in seqs:
            pred = classifier.best_char(s)
            if pred != ch:
                conf[(ch, pred)] = conf.get((ch, pred), 0) + 1
    top = sorted(conf.items(), key=lambda kv: -kv[1])[:12]
    print("Confusiones (real -> predicha: veces):",
          [f"{a}->{b}: {n}" for (a, b), n in top])

    out_path = os.path.join(here, "models_3d.json")
    ModelRepository(out_path).save(classifier)
    print(f"Modelo 3D guardado en: {out_path}")
