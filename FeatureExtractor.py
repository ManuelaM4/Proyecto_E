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