"""
Trainer.py
Pipeline de entrenamiento OFFLINE.
"""
import numpy as np
from HMMEngine import HMM, HMMClassifier

def build_sequences(raw, extractor, min_len=3):
    data = {}
    for ch, muestras in raw.items():
        seqs = [extractor.extract(m) for m in muestras]
        data[ch] = [s for s in seqs if len(s) >= min_len]
    return data

def split_dataset(data, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for ch, seqs in data.items():
        idx = rng.permutation(len(seqs))
        corte = int(train_ratio * len(seqs))
        train[ch] = [seqs[i] for i in idx[:corte]]
        test[ch] = [seqs[i] for i in idx[corte:]]
    return train, test

def train_classifier(data, n_states, n_symbols, n_iter=30, verbose=True):
    models = {}
    for ch, seqs in data.items():
        m = HMM(n_states, n_symbols, seed=1)
        m.baum_welch(seqs, n_iter=n_iter)
        models[ch] = m
        if verbose:
            print(f"  '{ch}' entrenado con {len(seqs)} muestras")
    return HMMClassifier(models)

def evaluate(classifier, test):
    ok = tot = 0
    por_letra = {}
    for ch, seqs in test.items():
        aciertos = sum(1 for s in seqs if classifier.best_char(s) == ch)
        por_letra[ch] = (aciertos, len(seqs))
        ok += aciertos
        tot += len(seqs)
    return ok / tot, por_letra