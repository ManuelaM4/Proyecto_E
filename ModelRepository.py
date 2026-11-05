"""
ModelRepository.py
Capa de persistencia.
"""
import json
from HMMEngine import HMM, HMMClassifier

class ModelRepository:
    def __init__(self, path):
        self.path = path

    def save(self, classifier):
        data = {c: m.to_dict() for c, m in classifier.models.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return HMMClassifier({c: HMM.from_dict(d) for c, d in data.items()})