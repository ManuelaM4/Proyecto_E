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
# NUEVO: Extractor para el Modo 3D
# ======================================================================
class Extractor3D(FeatureExtractor):
    """
    Extractor para el modo 3D.
    Convierte puntos (x, y, z) de la esfera a coordenadas esféricas (theta, phi)
    y luego extrae la dirección del movimiento angular (Δtheta, Δphi).
    """
    def __init__(self, k_dir=16, min_angle_dist=0.05):
        self.k_dir = k_dir
        self.min_angle_dist = min_angle_dist
        self.pen_up = k_dir

    @property
    def n_symbols(self):
        return self.k_dir + 1

    def _to_spherical(self, p):
        """Convierte un punto cartesiano (x,y,z) a coordenadas esféricas (theta, phi)."""
        x, y, z = p[0], p[1], p[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0: 
            return (0.0, 0.0)
        theta = np.arccos(np.clip(z / r, -1.0, 1.0)) # Ángulo polar (0 a pi)
        phi = np.arctan2(y, x)                       # Ángulo azimutal (-pi a pi)
        return (theta, phi)

    def _direction_symbol(self, p1, p2):
        t1, f1 = self._to_spherical(p1)
        t2, f2 = self._to_spherical(p2)
        
        # Diferencia angular
        dt = t2 - t1
        df = f2 - f1
        # Normalizar phi para evitar saltos en los polos
        if df > np.pi: df -= 2 * np.pi
        elif df < -np.pi: df += 2 * np.pi

        # Si el movimiento es muy pequeño, no se emite símbolo
        if np.sqrt(dt**2 + df**2) < self.min_angle_dist:
            return None

        # Cuantización del vector (dt, df) en sectores de dirección
        ang = np.arctan2(dt, df) % (2 * np.pi)
        sector = 2 * np.pi / self.k_dir
        return int(np.floor(ang / sector + 0.5)) % self.k_dir

    def _stroke_symbols(self, stroke):
        pts = np.asarray(stroke, dtype=float)
        if len(pts) < 2: 
            return []
        symbols = []
        for i in range(len(pts) - 1):
            sym = self._direction_symbol(pts[i], pts[i+1])
            if sym is not None:
                symbols.append(sym)
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
        # La proyección futura en 3D es compleja y se puede dejar para después.
        # Por ahora retornamos una lista vacía para que la interfaz no falle.
        return []