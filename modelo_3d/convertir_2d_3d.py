import json
import os
import numpy as np

print("[convertir] archivo cargado:", __file__, flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
RUTA_2D = os.path.join(ROOT, "modelo_2d", "glifo-dataset-2026-11-12.json")
OUT = os.path.join(BASE, "glifo-3d.json")
OUT_AUG = os.path.join(BASE, "glifo-3d-aug.json")

N_AUG = 5                 # copias aumentadas por muestra
ROT_MAX = 35.0            # grados de giro de la esfera en x e y
ROLL_MAX = 15.0           # grados de giro en el plano (z)
ESCALA = (0.8, 1.2)       # variación de tamaño
PX_POR_RADIO = 500.0      # escala FIJA: 500 px del dataset 2D = 1 radio de esfera
                          # (conserva el tamaño relativo entre letras, igual que el 2D)
SEED = 42


def rot_matrix(ax, ay, az):
    ax, ay, az = np.radians([ax, ay, az])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def plano_a_esfera(xy):
    """(N,2) en plano unitario (y arriba) -> (N,3) sobre la esfera."""
    r2 = (xy ** 2).sum(axis=1)
    fuera = r2 > 1.0
    xy = xy.copy()
    xy[fuera] /= np.sqrt(r2[fuera])[:, None]
    r2 = np.minimum(r2, 1.0)
    z = np.sqrt(np.maximum(0.0, 1.0 - r2))
    return np.column_stack([xy, z])


def muestra_a_3d(muestra, factor=1.0, R=None):
    trazos = [np.array([[p[0], p[1]] for p in t], dtype=float)
              for t in muestra if len(t) >= 2]
    if not trazos:
        return None
    todos = np.vstack(trazos)
    mn, mx = todos.min(axis=0), todos.max(axis=0)
    centro = (mn + mx) / 2.0
    if max(mx - mn) < 1e-6:
        return None
    s = factor / PX_POR_RADIO              # misma escala para todas las letras
    salida = []
    for t in trazos:
        xy = (t - centro) * s
        xy[:, 1] *= -1.0                       # y de pantalla (abajo) -> y arriba
        pts = plano_a_esfera(xy)
        if R is not None:
            pts = pts @ R.T
        salida.append([[round(float(a), 5) for a in p] for p in pts])
    return salida


def main():
    if not os.path.exists(RUTA_2D):
        print(f"No se encontró {RUTA_2D}")
        return
    with open(RUTA_2D, "r", encoding="utf-8") as f:
        data_2d = json.load(f)
    print(f"Dataset 2D: {RUTA_2D}", flush=True)
    print(f"Cargado dataset 2D con {len(data_2d)} letras.", flush=True)

    rng = np.random.default_rng(SEED)
    originales, aumentos = {}, {}

    for letra, muestras in data_2d.items():
        originales[letra], aumentos[letra] = [], []
        for muestra in muestras:
            base = muestra_a_3d(muestra)
            if base is None:
                continue
            copias = []
            for _ in range(N_AUG):
                R = rot_matrix(rng.uniform(-ROT_MAX, ROT_MAX),
                               rng.uniform(-ROT_MAX, ROT_MAX),
                               rng.uniform(-ROLL_MAX, ROLL_MAX))
                c = muestra_a_3d(muestra, factor=rng.uniform(*ESCALA), R=R)
                if c is not None:
                    copias.append(c)
            originales[letra].append(base)
            aumentos[letra].append(copias)      # mismo índice que originales

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(originales, f)
    with open(OUT_AUG, "w", encoding="utf-8") as f:
        json.dump(aumentos, f)
    print(f"Originales: {OUT}\nAumentos:   {OUT_AUG}")


if __name__ == "__main__":
    print("[convertir] iniciando main()...", flush=True)
    main()
    print("[convertir] terminado.", flush=True)
