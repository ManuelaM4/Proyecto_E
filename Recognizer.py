"""
Recognizer.py
Capa de aplicación.
"""
class Recognizer:
    def __init__(self, extractor, classifier):
        self.extractor = extractor
        self.classifier = classifier

    def posteriors(self, strokes):
        seq = self.extractor.extract(strokes)
        if not seq:
            return []
        return self.classifier.posteriors(seq)

    def best(self, strokes):
        post = self.posteriors(strokes)
        return post[0] if post else None

    def autocomplete(self, strokes):
        seq = self.extractor.extract(strokes)
        if not seq:
            return None
        char, future = self.classifier.project(seq)
        last = strokes[-1][-1]
        return char, self.extractor.decode(future, last)