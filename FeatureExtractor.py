"""
FeatureExtractor.py
Capa geométrica. Convierte trazos en secuencias de símbolos discretos y
símbolos de vuelta en puntos. Cada modo (2D, 3D) es una subclase que cumple
la misma interfaz, así el resto del sistema no distingue entre ellos.
"""
from abc import ABC, abstractmethod
import numpy as np


class FeatureExtractor(ABC):
    """Interfaz común de los extractores de características."""

    @property
    @abstractmethod
    def n_symbols(self):
        """Tamaño del alfabeto de símbolos."""

    @abstractmethod
    def extract(self, strokes):
        """Trazos -> lista de símbolos. Funciona también con trazos parciales."""

    @abstractmethod
    def decode(self, symbols, origin):
        """Símbolos futuros -> puntos, partiendo de 'origin'."""


class Extractor2D(FeatureExtractor):
    """
    Código de cadena con k_dir sectores angulares más un símbolo extra
    PEN_UP que indica que se levantó el lápiz entre dos trazos.
    """

    def __init__(self, k_dir=16, min_dist=40.0):
        self.k_dir = k_dir
        self.min_dist = min_dist
        self.pen_up = k_dir

    @property
    def n_symbols(self):
        return self.k_dir + 1

    def _direction_symbol(self, p, q):
        dx = q[0] - p[0]
        dy = -(q[1] - p[1])                       # el eje y de pantalla va hacia abajo
        ang = np.arctan2(dy, dx) % (2 * np.pi)
        sector = 2 * np.pi / self.k_dir
        return int(np.floor(ang / sector + 0.5)) % self.k_dir

    def _stroke_symbols(self, stroke):
        pts = np.asarray(stroke, dtype=float)[:, :2]
        if len(pts) < 2:
            return []
        symbols = []
        last = pts[0]
        for p in pts[1:]:
            if np.hypot(p[0] - last[0], p[1] - last[1]) >= self.min_dist:
                symbols.append(self._direction_symbol(last, p))
                last = p
        return symbols

    def extract(self, strokes):
        seq = []
        for stroke in strokes:
            s = self._stroke_symbols(stroke)
            if not s:
                continue
            if seq:
                seq.append(self.pen_up)
            seq.extend(s)
        return seq

    def decode(self, symbols, origin):
        """
        Reconstruye la trayectoria futura como una polilínea que parte de origin.
        Se detiene en el primer PEN_UP (no se sabe dónde empieza el siguiente trazo).
        """
        x, y = float(origin[0]), float(origin[1])
        points = []
        for s in symbols:
            if s == self.pen_up:
                break
            ang = s * 2 * np.pi / self.k_dir
            x += self.min_dist * np.cos(ang)
            y -= self.min_dist * np.sin(ang)
            points.append((x, y))
        return points


# ======================================================================
# Extractor para el Modo 3D
# ======================================================================
class Extractor3D(FeatureExtractor):
    """
    Extractor para el modo 3D.
    Para cada par de puntos sobre la esfera calcula el rumbo (heading) en el
    plano tangente local, tomando como "norte" la dirección "arriba de la
    pantalla" (eje +y). En el centro de la esfera coincide con el chain code 2D.
    Emite un símbolo solo cuando la distancia geodésica al último punto
    emitido supera min_angle_dist (acumulativo, igual que en 2D).
    """
    def __init__(self, k_dir=16, min_angle_dist=0.03):
        self.k_dir = k_dir
        self.min_angle_dist = min_angle_dist
        self.pen_up = k_dir

    @property
    def n_symbols(self):
        return self.k_dir + 1

    @staticmethod
    def _unit(p):
        p = np.asarray(p, dtype=float)[:3]
        n = np.linalg.norm(p)
        return p / n if n > 0 else np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _frame(p):
        """Base tangente local en p: (norte, este)."""
        up = np.array([0.0, 1.0, 0.0])
        n = up - np.dot(up, p) * p
        nn = np.linalg.norm(n)
        if nn < 1e-6:                       # justo en el polo superior/inferior
            n = np.array([0.0, 0.0, 1.0])
            n = n - np.dot(n, p) * p
            nn = np.linalg.norm(n)
        n = n / nn
        e = np.cross(n, p)
        return n, e

    def _direction_symbol(self, p, q):
        n, e = self._frame(p)
        d = q - p
        ang = np.arctan2(np.dot(d, n), np.dot(d, e)) % (2 * np.pi)
        sector = 2 * np.pi / self.k_dir
        return int(np.floor(ang / sector + 0.5)) % self.k_dir

    def _stroke_symbols(self, stroke):
        if len(stroke) < 2:
            return []
        pts = [self._unit(p) for p in stroke]
        symbols = []
        last = pts[0]
        for p in pts[1:]:
            dist = np.arccos(np.clip(np.dot(last, p), -1.0, 1.0))
            if dist >= self.min_angle_dist:
                symbols.append(self._direction_symbol(last, p))
                last = p
        return symbols

    def extract(self, strokes):
        seq = []
        for stroke in strokes:
            s = self._stroke_symbols(stroke)
            if not s:
                continue
            if seq:
                seq.append(self.pen_up)
            seq.extend(s)
        return seq

    def decode(self, symbols, origin):
        """
        Reconstruye la trayectoria futura caminando sobre la esfera:
        en cada paso avanza min_angle_dist en la dirección del símbolo.
        Devuelve puntos (x, y, z) unitarios. Se detiene en el primer PEN_UP.
        """
        p = self._unit(origin)
        points = []
        step = self.min_angle_dist
        for s in symbols:
            if s == self.pen_up:
                break
            n, e = self._frame(p)
            ang = s * 2 * np.pi / self.k_dir
            d = np.cos(ang) * e + np.sin(ang) * n
            p = p * np.cos(step) + d * np.sin(step)
            p = p / np.linalg.norm(p)
            points.append((float(p[0]), float(p[1]), float(p[2])))
        return points
