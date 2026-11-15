import os
import tkinter as tk
import customtkinter as ctk
import numpy as np
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont
import math

from Recognizer import Recognizer
from FeatureExtractor import Extractor2D, Extractor3D
from ModelRepository import ModelRepository

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_2D = os.path.join(BASE, "modelo_2d", "models_2d.json")
MODEL_3D = os.path.join(BASE, "modelo_3d", "models_3d.json")
MIN_DIST_2D = 15.0
MIN_ANGLE_3D = 0.03     
BG, CARD, TEXT, MUTED = "#f0f4f8", "#ffffff", "#1e293b", "#64748b"
ACCENT, ACCENT_SOFT, TILE, BAR_OFF = "#2563eb", "#60a5fa", "#e0f2fe", "#e2e8f0"
INK, SURFACE, SURFACE_EDGE, DANGER = "#0f172a", "#f8fafc", "#e2e8f0", "#ef4444"
SUCCESS, ROW_FIRST = "#10b981", "#d1fae5"

ICON_DIR = os.path.join(BASE, "assets", "iconos")
DATA_SIZE = 1000.0
BRUSH = 5
MIN_STEP = 2.0
FOLLOW = 0.55
UPDATE_MS = 200
TEMPERATURE = 4.0
TOP_N = 8

# --- GUÍAS DE TRAZO (generadas con generar_guias.py a partir del dataset de Glifo Captura) ---
# Cada letra es una lista de trazos EN EL ORDEN Y SENTIDO en que fueron capturados.
# Cada trazo es una lista de puntos (x, y) en un cuadro de 0 a 100 (y hacia abajo).
# Un trazo de un solo punto es un punto suelto (punto de la i y la j).
TRAZOS_GUIA = {
    "a": [
        [(63.1, 23.1), (59.5, 18.4), (54.5, 15.2), (48.8, 13.3), (42.9, 12.4), (37.0, 12.6), (31.2, 13.9), (25.9, 16.5), (21.4, 20.5), (17.8, 25.2), (14.9, 30.5), (12.7, 36.0), (11.3, 41.8), (10.4, 47.7), (10.0, 53.7), (10.1, 59.6), (10.6, 65.6), (12.1, 71.4), (14.8, 76.7), (18.8, 81.1), (23.7, 84.5), (29.2, 86.6), (35.1, 87.6), (41.1, 87.4), (46.9, 86.1), (52.2, 83.4), (56.7, 79.6), (60.4, 74.9), (63.2, 69.6), (65.2, 64.0), (66.6, 58.2), (67.4, 52.3)],
        [(65.6, 20.2), (65.4, 22.8), (65.4, 25.3), (65.4, 27.8), (65.4, 30.3), (65.5, 32.8), (65.5, 35.3), (65.6, 37.8), (65.7, 40.3), (65.8, 42.8), (66.0, 45.3), (66.1, 47.8), (66.3, 50.3), (66.5, 52.8), (66.8, 55.3), (67.2, 57.8), (67.6, 60.3), (68.1, 62.7), (68.7, 65.2), (69.3, 67.6), (70.1, 70.0), (71.0, 72.3), (72.0, 74.6), (73.2, 76.8), (74.7, 78.9), (76.4, 80.7), (78.3, 82.3), (80.5, 83.5), (82.9, 84.3), (85.4, 84.6), (87.9, 84.2), (90.0, 82.9)],
    ],
    "b": [
        [(32.4, 10.0), (32.8, 16.6), (33.0, 23.3), (33.1, 29.9), (33.0, 36.5), (33.0, 43.2), (33.0, 49.8), (32.9, 56.4), (32.8, 63.1), (32.8, 69.7), (32.7, 76.4), (32.6, 83.0), (32.2, 85.6), (32.4, 78.9), (33.2, 72.3), (34.8, 65.9), (37.6, 59.9), (41.9, 54.9), (47.7, 51.8), (54.2, 51.7), (59.9, 55.0), (64.1, 60.1), (66.6, 66.2), (67.8, 72.7), (67.3, 79.3), (64.6, 85.3), (59.4, 89.1), (52.9, 90.0), (46.6, 88.0), (41.0, 84.4), (36.7, 79.4), (33.2, 73.7)],
    ],
    "c": [
        [(75.7, 18.8), (71.6, 15.0), (66.7, 12.3), (61.3, 10.7), (55.7, 10.0), (50.1, 10.1), (44.6, 11.0), (39.4, 13.1), (34.7, 16.2), (30.8, 20.2), (27.5, 24.8), (24.9, 29.7), (23.0, 35.0), (21.7, 40.5), (20.9, 46.0), (20.6, 51.6), (20.6, 57.3), (21.1, 62.9), (22.2, 68.3), (24.3, 73.6), (27.3, 78.3), (31.1, 82.4), (35.7, 85.7), (40.8, 88.0), (46.2, 89.5), (51.8, 90.0), (57.4, 89.7), (62.8, 88.4), (67.9, 86.1), (72.4, 82.6), (76.2, 78.5), (79.4, 73.9)],
    ],
    "d": [
        [(62.8, 52.5), (60.3, 51.4), (57.6, 50.6), (54.9, 50.2), (52.1, 50.1), (49.3, 50.4), (46.6, 51.0), (44.0, 52.0), (41.6, 53.4), (39.5, 55.1), (37.7, 57.3), (36.2, 59.6), (35.2, 62.2), (34.5, 64.9), (34.1, 67.7), (34.0, 70.4), (34.2, 73.2), (34.7, 76.0), (35.6, 78.6), (36.9, 81.0), (38.6, 83.3), (40.6, 85.2), (42.8, 86.9), (45.2, 88.2), (47.9, 89.1), (50.6, 89.7), (53.3, 90.0), (56.1, 90.0), (58.9, 89.6), (61.5, 88.6), (63.6, 86.8), (65.0, 84.4)],
        [(63.9, 10.0), (63.7, 12.6), (63.6, 15.1), (63.6, 17.7), (63.5, 20.2), (63.4, 22.8), (63.4, 25.3), (63.4, 27.9), (63.4, 30.5), (63.4, 33.0), (63.4, 35.6), (63.4, 38.1), (63.4, 40.7), (63.5, 43.2), (63.5, 45.8), (63.6, 48.4), (63.6, 50.9), (63.7, 53.5), (63.9, 56.0), (64.0, 58.6), (64.1, 61.1), (64.3, 63.7), (64.4, 66.2), (64.5, 68.8), (64.6, 71.3), (64.7, 73.9), (64.9, 76.5), (65.0, 79.0), (65.2, 81.6), (65.4, 84.1), (65.7, 86.6), (66.0, 89.2)],
    ],
    "e": [
        [(13.6, 38.8), (22.1, 38.9), (30.6, 39.0), (39.1, 39.1), (47.6, 39.2), (56.2, 39.3), (64.7, 39.1), (73.1, 37.9), (79.6, 33.1), (79.0, 25.0), (73.8, 18.3), (66.6, 13.7), (58.6, 10.9), (50.2, 10.0), (41.7, 10.8), (33.8, 13.7), (26.9, 18.7), (21.6, 25.3), (18.1, 33.1), (16.2, 41.4), (15.7, 49.8), (16.3, 58.3), (18.6, 66.5), (23.2, 73.6), (29.5, 79.3), (36.8, 83.6), (44.8, 86.5), (53.1, 88.5), (61.5, 89.6), (70.0, 90.0), (78.5, 89.1), (86.4, 86.1)],
    ],
    "f": [
        [(71.0, 16.5), (69.7, 13.6), (67.4, 11.4), (64.4, 10.3), (61.2, 10.0), (58.1, 10.6), (55.2, 12.1), (52.9, 14.2), (51.1, 16.9), (49.8, 19.8), (48.7, 22.9), (48.0, 26.0), (47.4, 29.2), (47.0, 32.3), (46.6, 35.5), (46.3, 38.7), (46.1, 42.0), (45.9, 45.2), (45.7, 48.4), (45.5, 51.6), (45.3, 54.8), (45.1, 58.0), (44.9, 61.2), (44.8, 64.4), (44.6, 67.6), (44.4, 70.8), (44.1, 74.0), (43.9, 77.2), (43.5, 80.4), (43.1, 83.6), (42.7, 86.8), (42.2, 90.0)],
        [(29.0, 46.5), (30.1, 46.3), (31.2, 46.2), (32.3, 46.1), (33.4, 46.0), (34.5, 45.9), (35.6, 45.9), (36.7, 45.8), (37.8, 45.7), (38.9, 45.6), (40.0, 45.5), (41.1, 45.4), (42.2, 45.4), (43.3, 45.3), (44.4, 45.2), (45.5, 45.2), (46.6, 45.1), (47.7, 45.0), (48.8, 45.0), (49.9, 44.9), (51.0, 44.9), (52.1, 44.8), (53.2, 44.8), (54.3, 44.7), (55.4, 44.7), (56.5, 44.7), (57.6, 44.6), (58.7, 44.6), (59.8, 44.6), (60.9, 44.5), (62.0, 44.4), (63.1, 44.3)],
    ],
    "g": [
        [(53.6, 10.3), (49.7, 10.0), (45.9, 10.5), (42.4, 12.2), (39.6, 14.8), (37.4, 18.0), (35.9, 21.6), (34.9, 25.4), (34.4, 29.2), (34.2, 33.1), (34.3, 37.0), (34.9, 40.8), (36.1, 44.5), (38.1, 47.8), (40.8, 50.6), (44.2, 52.5), (47.9, 53.5), (51.8, 53.7), (55.6, 53.0), (58.9, 51.1), (61.5, 48.2), (63.2, 44.7), (64.0, 41.0), (64.4, 37.1), (64.4, 33.2), (64.1, 29.3), (63.4, 25.5), (62.3, 21.8), (60.7, 18.3), (58.4, 15.1), (55.7, 12.4), (52.6, 10.0)],
        [(56.0, 11.3), (58.5, 13.8), (60.5, 16.7), (61.9, 19.9), (62.9, 23.3), (63.6, 26.8), (64.2, 30.3), (64.6, 33.8), (65.0, 37.4), (65.3, 40.9), (65.6, 44.4), (65.7, 48.0), (65.8, 51.5), (65.8, 55.1), (65.7, 58.6), (65.6, 62.2), (65.3, 65.7), (65.0, 69.2), (64.5, 72.7), (63.9, 76.2), (63.1, 79.7), (61.7, 83.0), (59.8, 85.9), (57.0, 88.1), (53.8, 89.5), (50.3, 90.0), (46.8, 89.3), (43.8, 87.4), (41.5, 84.8), (39.8, 81.7), (38.6, 78.3), (37.8, 74.9)],
    ],
    "h": [
        [(36.5, 10.0), (37.0, 15.2), (37.4, 20.3), (37.7, 25.5), (37.8, 30.7), (37.9, 35.9), (38.0, 41.1), (38.0, 46.3), (38.1, 51.5), (38.1, 56.7), (38.2, 61.8), (38.2, 67.0), (38.2, 72.2), (38.2, 77.4), (38.2, 82.6), (38.1, 86.4), (38.1, 81.2), (38.2, 76.1), (38.4, 70.9), (38.8, 65.7), (39.7, 60.6), (41.6, 55.8), (45.6, 52.6), (50.6, 52.6), (55.0, 55.2), (57.9, 59.5), (59.6, 64.4), (60.6, 69.4), (61.3, 74.6), (61.9, 79.8), (62.6, 84.9), (63.5, 90.0)],
    ],
    "i": [
        [(50.0, 24.5), (50.0, 26.6), (50.0, 28.7), (50.1, 30.8), (50.1, 32.9), (50.2, 35.0), (50.2, 37.2), (50.2, 39.3), (50.2, 41.4), (50.2, 43.5), (50.2, 45.6), (50.2, 47.7), (50.3, 49.8), (50.2, 52.0), (50.2, 54.1), (50.2, 56.2), (50.2, 58.3), (50.2, 60.4), (50.2, 62.5), (50.1, 64.6), (50.1, 66.7), (50.1, 68.9), (50.1, 71.0), (50.1, 73.1), (50.1, 75.2), (50.1, 77.3), (50.1, 79.4), (50.0, 81.5), (50.0, 83.7), (50.0, 85.8), (50.0, 87.9), (50.0, 90.0)],
        [(49.7, 10.0)],
    ],
    "j": [
        [(58.4, 20.4), (58.8, 23.3), (59.2, 26.3), (59.5, 29.3), (59.8, 32.2), (60.0, 35.2), (60.1, 38.1), (60.3, 41.1), (60.4, 44.1), (60.4, 47.1), (60.4, 50.0), (60.4, 53.0), (60.3, 56.0), (60.2, 58.9), (60.0, 61.9), (59.8, 64.9), (59.5, 67.8), (59.2, 70.8), (58.8, 73.7), (58.3, 76.6), (57.7, 79.5), (56.8, 82.4), (55.5, 85.1), (53.8, 87.5), (51.5, 89.3), (48.6, 90.0), (45.7, 89.5), (43.3, 87.8), (41.5, 85.5), (40.4, 82.7), (39.8, 79.8), (39.6, 76.9)],
        [(57.0, 10.0)],
    ],
    "k": [
        [(31.5, 10.0), (31.7, 12.6), (31.8, 15.2), (31.9, 17.7), (32.0, 20.3), (32.1, 22.9), (32.2, 25.5), (32.2, 28.1), (32.3, 30.6), (32.3, 33.2), (32.3, 35.8), (32.4, 38.4), (32.4, 41.0), (32.4, 43.5), (32.4, 46.1), (32.4, 48.7), (32.3, 51.3), (32.3, 53.9), (32.3, 56.4), (32.3, 59.0), (32.3, 61.6), (32.3, 64.2), (32.3, 66.8), (32.3, 69.4), (32.3, 71.9), (32.3, 74.5), (32.3, 77.1), (32.3, 79.7), (32.3, 82.3), (32.4, 84.8), (32.4, 87.4), (32.4, 90.0)],
        [(63.1, 23.0), (60.9, 25.0), (58.8, 26.9), (56.6, 29.0), (54.5, 31.0), (52.5, 33.1), (50.4, 35.2), (48.4, 37.3), (46.4, 39.5), (44.4, 41.6), (42.4, 43.8), (40.4, 45.9), (38.5, 48.2), (36.6, 50.4), (35.2, 53.0), (35.0, 55.9), (36.5, 58.3), (38.7, 60.3), (40.9, 62.2), (43.1, 64.1), (45.3, 66.1), (47.5, 68.1), (49.6, 70.1), (51.8, 72.1), (54.0, 74.0), (56.1, 76.0), (58.3, 78.1), (60.3, 80.1), (62.4, 82.2), (64.4, 84.4), (66.4, 86.5), (68.5, 88.6)],
    ],
    "l": [
        [(43.9, 10.0), (43.5, 12.9), (43.1, 15.7), (42.8, 18.6), (42.5, 21.4), (42.3, 24.3), (42.2, 27.2), (42.0, 30.1), (41.9, 32.9), (41.9, 35.8), (41.8, 38.7), (41.8, 41.6), (41.8, 44.5), (41.9, 47.4), (41.9, 50.2), (42.0, 53.1), (42.0, 56.0), (42.1, 58.9), (42.3, 61.8), (42.5, 64.6), (42.7, 67.5), (43.0, 70.4), (43.3, 73.2), (43.7, 76.1), (44.2, 78.9), (44.9, 81.7), (45.8, 84.4), (47.4, 86.8), (49.8, 88.5), (52.4, 89.5), (55.3, 89.9), (58.2, 90.0)],
    ],
    "m": [
        [(10.0, 15.2), (10.9, 25.5), (11.4, 35.8), (11.8, 46.1), (12.0, 56.4), (12.2, 66.7), (12.1, 77.0), (11.8, 70.0), (12.8, 59.8), (14.5, 49.6), (17.2, 39.7), (21.7, 30.5), (28.9, 23.2), (38.7, 22.6), (45.4, 30.0), (48.4, 39.8), (49.9, 50.0), (50.6, 60.3), (50.9, 70.6), (50.2, 69.2), (50.7, 58.9), (52.1, 48.7), (54.5, 38.6), (59.1, 29.5), (67.1, 23.3), (76.6, 26.0), (82.7, 34.1), (85.9, 43.9), (87.9, 54.0), (89.1, 64.2), (89.8, 74.5), (90.0, 84.8)],
    ],
    "n": [
        [(22.4, 10.0), (22.9, 17.9), (23.3, 25.9), (23.6, 33.8), (23.9, 41.7), (24.1, 49.7), (24.3, 57.6), (24.5, 65.5), (24.6, 73.5), (24.7, 81.4), (24.7, 78.9), (24.7, 71.0), (25.0, 63.1), (25.5, 55.2), (26.6, 47.3), (28.1, 39.5), (30.5, 31.9), (33.9, 24.8), (38.9, 18.6), (45.5, 14.4), (53.3, 13.2), (60.9, 15.0), (66.9, 20.1), (70.5, 27.1), (72.8, 34.7), (74.2, 42.5), (75.1, 50.4), (75.8, 58.3), (76.3, 66.2), (76.8, 74.1), (77.2, 82.1), (77.6, 90.0)],
    ],
    "o": [
        [(52.6, 10.0), (44.7, 10.1), (37.0, 11.6), (30.2, 15.6), (24.6, 21.2), (20.4, 27.9), (17.2, 35.1), (15.1, 42.7), (13.9, 50.6), (13.8, 58.5), (14.8, 66.3), (17.5, 73.7), (22.0, 80.2), (28.2, 85.1), (35.5, 88.1), (43.2, 89.7), (51.1, 90.0), (58.9, 88.9), (66.4, 86.3), (72.9, 81.8), (78.1, 75.9), (82.0, 69.0), (84.5, 61.5), (85.9, 53.7), (86.2, 45.8), (85.4, 38.0), (83.4, 30.3), (79.5, 23.5), (74.1, 17.7), (67.4, 13.5), (59.9, 11.4), (52.0, 11.5)],
    ],
    "p": [
        [(36.0, 10.0), (36.1, 12.6), (36.3, 15.2), (36.3, 17.8), (36.4, 20.3), (36.5, 22.9), (36.7, 25.5), (36.8, 28.1), (36.9, 30.7), (37.0, 33.3), (37.2, 35.9), (37.3, 38.5), (37.5, 41.0), (37.6, 43.6), (37.8, 46.2), (37.9, 48.8), (38.0, 51.4), (38.1, 54.0), (38.2, 56.6), (38.2, 59.1), (38.2, 61.7), (38.2, 64.3), (38.2, 66.9), (38.2, 69.5), (38.1, 72.1), (38.0, 74.7), (37.9, 77.3), (37.8, 79.9), (37.7, 82.4), (37.6, 85.0), (37.5, 87.6), (37.1, 90.0)],
        [(36.9, 16.0), (38.8, 14.3), (41.0, 12.9), (43.4, 12.0), (46.0, 11.6), (48.6, 11.6), (51.2, 12.1), (53.6, 13.0), (55.8, 14.5), (57.6, 16.3), (59.2, 18.4), (60.5, 20.6), (61.6, 23.0), (62.4, 25.5), (63.1, 28.0), (63.6, 30.6), (63.9, 33.2), (64.0, 35.8), (63.9, 38.4), (63.6, 41.0), (62.9, 43.5), (61.8, 45.9), (60.4, 48.0), (58.6, 50.0), (56.6, 51.6), (54.3, 52.9), (51.8, 53.8), (49.3, 54.4), (46.7, 54.7), (44.1, 54.7), (41.5, 54.3), (39.0, 53.5)],
    ],
    "q": [
        [(61.3, 16.9), (58.7, 14.0), (55.5, 11.8), (51.8, 10.5), (47.9, 10.0), (43.9, 10.1), (40.1, 10.9), (36.5, 12.6), (33.4, 15.1), (30.9, 18.1), (29.1, 21.6), (27.8, 25.3), (27.0, 29.2), (26.7, 33.1), (26.9, 37.1), (27.7, 40.9), (29.4, 44.5), (32.1, 47.3), (35.5, 49.3), (39.3, 50.4), (43.2, 50.6), (47.1, 50.2), (50.8, 48.7), (54.0, 46.4), (56.6, 43.5), (58.6, 40.1), (60.0, 36.4), (61.0, 32.6), (61.7, 28.7), (62.1, 24.8), (62.3, 20.8), (62.4, 16.9)],
        [(62.2, 11.4), (62.3, 13.9), (62.4, 16.5), (62.4, 19.0), (62.4, 21.5), (62.4, 24.1), (62.3, 26.6), (62.2, 29.1), (62.2, 31.7), (62.0, 34.2), (61.9, 36.7), (61.7, 39.3), (61.5, 41.8), (61.4, 44.3), (61.2, 46.9), (61.1, 49.4), (61.0, 51.9), (60.9, 54.5), (60.8, 57.0), (60.7, 59.6), (60.6, 62.1), (60.5, 64.6), (60.4, 67.2), (60.4, 69.7), (60.3, 72.2), (60.2, 74.8), (60.2, 77.3), (60.2, 79.9), (60.1, 82.4), (60.0, 84.9), (60.0, 87.5), (59.9, 90.0)],
        [(45.7, 73.6), (46.6, 73.5), (47.5, 73.5), (48.4, 73.4), (49.3, 73.4), (50.2, 73.3), (51.1, 73.3), (52.0, 73.3), (52.8, 73.3), (53.7, 73.3), (54.6, 73.2), (55.5, 73.2), (56.4, 73.2), (57.3, 73.2), (58.2, 73.1), (59.1, 73.1), (60.0, 73.0), (60.9, 73.0), (61.7, 73.0), (62.6, 72.9), (63.5, 72.8), (64.4, 72.8), (65.3, 72.7), (66.2, 72.7), (67.1, 72.6), (68.0, 72.5), (68.8, 72.4), (69.7, 72.3), (70.6, 72.3), (71.5, 72.2), (72.4, 72.0), (73.3, 71.9)],
    ],
    "r": [
        [(26.7, 10.0), (27.3, 16.1), (27.7, 22.2), (28.0, 28.3), (28.3, 34.5), (28.5, 40.6), (28.6, 46.7), (28.7, 52.8), (28.8, 59.0), (28.9, 65.1), (29.0, 71.2), (29.0, 77.3), (29.1, 83.5), (29.2, 89.6), (29.2, 90.0), (29.2, 83.9), (29.3, 77.7), (29.6, 71.6), (29.9, 65.5), (30.2, 59.4), (30.8, 53.3), (31.6, 47.2), (32.8, 41.2), (34.5, 35.3), (37.0, 29.7), (40.4, 24.6), (44.6, 20.2), (49.5, 16.6), (55.2, 14.3), (61.2, 13.3), (67.3, 13.5), (73.3, 14.8)],
    ],
    "s": [
        [(86.7, 12.3), (79.9, 10.8), (72.9, 10.0), (66.0, 10.0), (59.0, 10.6), (52.2, 11.9), (45.5, 14.0), (39.4, 17.2), (34.9, 22.4), (34.3, 29.2), (37.6, 35.1), (43.0, 39.5), (49.1, 43.0), (55.3, 46.1), (61.7, 49.0), (68.0, 52.0), (74.0, 55.5), (79.3, 60.0), (82.7, 66.0), (83.6, 72.8), (81.5, 79.4), (76.7, 84.4), (70.6, 87.6), (63.8, 89.3), (56.9, 90.0), (50.0, 89.8), (43.1, 88.7), (36.3, 87.0), (29.8, 84.7), (23.5, 81.7), (17.8, 77.6), (13.3, 72.3)],
    ],
    "t": [
        [(46.8, 10.0), (46.3, 13.2), (45.8, 16.4), (45.3, 19.6), (44.8, 22.9), (44.4, 26.1), (44.0, 29.3), (43.7, 32.6), (43.5, 35.8), (43.3, 39.1), (43.2, 42.3), (43.1, 45.6), (43.2, 48.8), (43.2, 52.1), (43.4, 55.3), (43.6, 58.6), (43.8, 61.8), (44.2, 65.0), (44.6, 68.3), (45.1, 71.5), (45.8, 74.7), (46.6, 77.8), (47.8, 80.8), (49.3, 83.7), (51.3, 86.3), (53.8, 88.3), (56.8, 89.5), (60.0, 90.0), (63.3, 89.9), (66.4, 89.1), (69.2, 87.4), (71.0, 84.7)],
        [(29.0, 47.2), (30.1, 47.2), (31.3, 47.3), (32.4, 47.3), (33.5, 47.4), (34.7, 47.4), (35.8, 47.5), (36.9, 47.5), (38.1, 47.6), (39.2, 47.6), (40.3, 47.7), (41.5, 47.8), (42.6, 47.9), (43.7, 48.0), (44.9, 48.1), (46.0, 48.2), (47.1, 48.3), (48.2, 48.4), (49.4, 48.4), (50.5, 48.5), (51.6, 48.6), (52.8, 48.7), (53.9, 48.8), (55.0, 48.9), (56.2, 49.0), (57.3, 49.1), (58.4, 49.2), (59.5, 49.3), (60.7, 49.4), (61.8, 49.5), (62.9, 49.6), (64.1, 49.7)],
    ],
    "u": [
        [(19.9, 10.0), (19.8, 18.0), (19.9, 26.0), (20.0, 33.9), (20.2, 41.9), (20.6, 49.9), (21.4, 57.8), (22.8, 65.7), (25.3, 73.3), (29.2, 80.2), (34.9, 85.6), (42.3, 88.4), (50.2, 87.7), (56.9, 83.6), (61.4, 77.0), (64.3, 69.6), (66.1, 61.9), (67.2, 54.0), (67.8, 46.0), (68.2, 38.0), (68.3, 30.0), (68.2, 22.1), (67.2, 21.4), (66.9, 29.4), (66.9, 37.4), (67.1, 45.4), (67.4, 53.3), (68.1, 61.3), (69.3, 69.2), (71.4, 76.9), (75.0, 84.0), (80.2, 90.0)],
    ],
    "v": [
        [(16.1, 15.2), (17.8, 20.4), (19.8, 25.5), (22.0, 30.6), (24.3, 35.6), (26.6, 40.6), (28.9, 45.6), (31.3, 50.6), (33.6, 55.6), (36.0, 60.6), (38.3, 65.6), (40.6, 70.6), (43.0, 75.6), (45.3, 80.5), (47.7, 85.5), (50.7, 90.0), (54.1, 87.0), (55.8, 81.8), (57.4, 76.5), (59.1, 71.2), (60.8, 66.0), (62.6, 60.8), (64.4, 55.6), (66.3, 50.4), (68.3, 45.3), (70.4, 40.2), (72.5, 35.1), (74.7, 30.0), (76.9, 25.0), (79.1, 20.0), (81.5, 15.0), (83.9, 10.0)],
    ],
    "w": [
        [(10.0, 18.4), (12.8, 24.9), (15.7, 31.4), (18.4, 37.9), (21.0, 44.5), (23.6, 51.1), (26.1, 57.7), (28.7, 64.3), (31.4, 70.8), (34.4, 77.2), (38.9, 82.3), (43.0, 77.3), (45.0, 70.5), (46.7, 63.7), (48.5, 56.8), (50.7, 50.1), (55.1, 48.9), (57.5, 55.6), (59.7, 62.3), (62.1, 69.0), (64.9, 75.5), (68.9, 81.2), (74.2, 79.2), (76.1, 72.5), (77.4, 65.5), (78.6, 58.5), (79.9, 51.6), (81.5, 44.7), (83.2, 37.8), (85.1, 31.0), (87.4, 24.3), (90.0, 17.7)],
    ],
    "x": [
        [(19.1, 13.5), (20.6, 16.3), (22.2, 19.0), (23.9, 21.7), (25.7, 24.3), (27.5, 26.9), (29.4, 29.5), (31.3, 32.0), (33.2, 34.6), (35.2, 37.1), (37.1, 39.6), (39.1, 42.1), (41.1, 44.6), (43.1, 47.1), (45.1, 49.5), (47.1, 52.0), (49.2, 54.4), (51.2, 56.9), (53.3, 59.3), (55.3, 61.7), (57.4, 64.1), (59.5, 66.5), (61.6, 68.9), (63.7, 71.3), (65.8, 73.7), (68.0, 76.0), (70.1, 78.4), (72.3, 80.7), (74.4, 83.0), (76.6, 85.3), (78.8, 87.7), (80.9, 90.0)],
        [(22.4, 85.8), (24.0, 83.2), (25.4, 80.5), (26.8, 77.8), (28.2, 75.1), (29.6, 72.4), (30.9, 69.7), (32.3, 67.0), (33.7, 64.3), (35.1, 61.6), (36.5, 58.9), (38.0, 56.3), (39.5, 53.7), (41.1, 51.1), (42.7, 48.5), (44.4, 46.0), (46.1, 43.5), (47.9, 41.0), (49.7, 38.6), (51.6, 36.2), (53.5, 33.8), (55.4, 31.5), (57.4, 29.2), (59.5, 27.0), (61.6, 24.8), (63.7, 22.7), (65.9, 20.5), (68.1, 18.4), (70.3, 16.3), (72.5, 14.3), (74.6, 12.2), (76.8, 10.0)],
    ],
    "y": [
        [(30.9, 10.0), (31.8, 11.5), (32.7, 13.0), (33.6, 14.5), (34.4, 16.0), (35.3, 17.5), (36.2, 19.0), (37.0, 20.5), (37.9, 22.0), (38.8, 23.5), (39.7, 25.0), (40.5, 26.5), (41.4, 28.0), (42.3, 29.4), (43.2, 30.9), (44.1, 32.4), (44.9, 33.9), (45.8, 35.4), (46.7, 36.9), (47.6, 38.4), (48.4, 39.9), (49.3, 41.4), (50.2, 42.9), (51.1, 44.4), (52.0, 45.9), (52.9, 47.4), (53.8, 48.9), (54.7, 50.4), (55.6, 51.8), (56.5, 53.3), (57.4, 54.8), (58.5, 56.1)],
        [(71.9, 13.4), (71.4, 16.4), (70.8, 19.3), (70.1, 22.3), (69.4, 25.2), (68.6, 28.0), (67.7, 30.9), (66.9, 33.8), (66.0, 36.7), (65.1, 39.5), (64.1, 42.4), (63.1, 45.2), (62.1, 48.0), (61.0, 50.8), (59.9, 53.6), (58.8, 56.4), (57.6, 59.1), (56.4, 61.9), (55.1, 64.6), (53.7, 67.3), (52.3, 69.9), (50.8, 72.5), (49.2, 75.0), (47.5, 77.5), (45.7, 79.9), (43.7, 82.2), (41.6, 84.3), (39.3, 86.2), (36.7, 87.8), (34.0, 89.0), (31.1, 89.8), (28.1, 90.0)],
    ],
    "z": [
        [(10.0, 12.5), (18.2, 12.3), (26.3, 12.3), (34.5, 12.2), (42.7, 12.2), (50.9, 12.2), (59.0, 12.1), (67.2, 12.0), (75.4, 12.3), (81.0, 16.0), (76.0, 22.4), (70.1, 28.0), (64.0, 33.4), (57.9, 38.9), (51.8, 44.4), (45.8, 49.9), (39.7, 55.4), (33.7, 60.9), (27.8, 66.6), (21.9, 72.2), (16.1, 78.0), (11.0, 84.3), (16.4, 87.9), (24.6, 87.8), (32.8, 87.6), (41.0, 87.5), (49.1, 87.5), (57.3, 87.5), (65.5, 87.5), (73.7, 87.6), (81.8, 87.7), (90.0, 88.0)],
    ],
    "ñ": [
        [(28.8, 32.5), (29.4, 38.5), (29.9, 44.5), (30.2, 50.5), (30.5, 56.6), (30.7, 62.6), (30.9, 68.6), (31.1, 74.7), (31.2, 80.7), (31.3, 86.7), (31.1, 82.4), (31.0, 76.4), (31.2, 70.4), (31.6, 64.3), (32.4, 58.3), (33.5, 52.4), (35.3, 46.6), (37.8, 41.2), (41.7, 36.6), (46.8, 33.5), (52.7, 32.5), (58.5, 33.6), (63.3, 37.3), (66.3, 42.5), (68.2, 48.2), (69.5, 54.1), (70.6, 60.0), (71.5, 66.0), (72.3, 72.0), (73.0, 78.0), (73.5, 84.0), (73.9, 90.0)],
        [(26.1, 20.9), (26.7, 19.3), (27.5, 17.8), (28.5, 16.4), (29.5, 15.0), (30.6, 13.7), (31.8, 12.6), (33.2, 11.6), (34.8, 10.9), (36.4, 10.5), (38.1, 10.4), (39.8, 10.6), (41.4, 11.0), (43.0, 11.7), (44.5, 12.4), (45.9, 13.3), (47.3, 14.3), (48.7, 15.3), (50.1, 16.1), (51.6, 16.9), (53.2, 17.5), (54.9, 17.9), (56.6, 18.1), (58.2, 17.9), (59.9, 17.5), (61.4, 16.8), (62.8, 15.9), (64.2, 14.8), (65.5, 13.7), (66.7, 12.6), (67.8, 11.3), (68.9, 10.0)],
    ],
}

class GuiaEstatica:
    """Guía estática de trazos para una letra elegida.

    - Cada trazo tiene su propio color y un círculo numerado donde EMPIEZA (1, 2, 3...).
    - Las flechitas a lo largo del trazo y la flecha grande del final indican el SENTIDO.
    - La línea punteada gris une el final de un trazo con el inicio del siguiente:
      ahí hay que LEVANTAR EL LÁPIZ.
    Las coordenadas de TRAZOS_GUIA van en un cuadro de 0 a 100 (y hacia abajo).
    """
    W, H, PAD = 240, 200, 8
    COLORES = ["#2563eb", "#f97316", "#10b981", "#a855f7", "#ef4444"]
    LIFT = "#94a3b8"

    def __init__(self, canvas):
        self.canvas = canvas
        lado = min(self.W, self.H) - 2 * self.PAD
        self.k = lado / 100.0                      # escala UNIFORME: la letra no se deforma
        self.ox = (self.W - 100 * self.k) / 2
        self.oy = (self.H - 100 * self.k) / 2

    def _xy(self, p):
        return self.ox + p[0] * self.k, self.oy + p[1] * self.k

    @staticmethod
    def _densificar(pts, paso):
        """Puntos a distancia constante sobre la poligonal (no redondea esquinas)."""
        if len(pts) < 2:
            return [tuple(pts[0])]
        out, acum = [tuple(pts[0])], 0.0
        for a, b in zip(pts, pts[1:]):
            ax, ay = a
            bx, by = b
            seg = math.hypot(bx - ax, by - ay)
            while seg > 1e-9 and acum + seg >= paso:
                f = (paso - acum) / seg
                ax, ay = ax + (bx - ax) * f, ay + (by - ay) * f
                out.append((ax, ay))
                seg = math.hypot(bx - ax, by - ay)
                acum = 0.0
            acum += seg
        if math.hypot(out[-1][0] - pts[-1][0], out[-1][1] - pts[-1][1]) > 1e-6:
            out.append(tuple(pts[-1]))
        return out

    def _flecha(self, p, ang, color, L=9, w=0.55, relleno=None):
        x, y = self._xy(p)
        self.canvas.create_polygon(
            x + L * math.cos(ang), y + L * math.sin(ang),
            x + L * 0.6 * math.cos(ang + math.pi - w) , y + L * 0.6 * math.sin(ang + math.pi - w),
            x + L * 0.6 * math.cos(ang + math.pi + w), y + L * 0.6 * math.sin(ang + math.pi + w),
            fill=relleno or color, outline="white", width=1, tags="flecha")

    def _fondo(self):
        for f in (0.1, 0.5, 0.9):
            y = self.oy + 100 * self.k * f
            self.canvas.create_line(self.ox, y, self.ox + 100 * self.k, y,
                                    fill="#e2e8f0", width=1, dash=(2, 4))

    def mostrar(self, letra):
        c = self.canvas
        c.delete("all")
        self._fondo()
        trazos = TRAZOS_GUIA.get(letra, [])
        if not trazos:
            c.create_text(self.W / 2, self.H / 2, text=letra, font=("Arial", 80, "bold"), fill="#cbd5e1")
            return
        colores = [self.COLORES[i % len(self.COLORES)] for i in range(len(trazos))]

        # 1) línea punteada = levantar el lápiz (del final de un trazo al inicio del siguiente)
        for t0, t1 in zip(trazos, trazos[1:]):
            x0, y0 = self._xy(t0[-1])
            x1, y1 = self._xy(t1[0])
            c.create_line(x0, y0, x1, y1, fill=self.LIFT, width=2, dash=(4, 4), tags="lift")

        # 2) trazos con flechas de sentido
        for t, col in zip(trazos, colores):
            if len(t) == 1:                                    # punto suelto (i, j)
                x, y = self._xy(t[0])
                c.create_oval(x - 6, y - 6, x + 6, y + 6, fill=col, outline="white", width=2, tags="trazo")
                continue
            c.create_line(*[v for p in t for v in self._xy(p)], fill=col, width=6,
                          capstyle="round", joinstyle="round", tags="trazo")
            d = self._densificar(t, 1.0)
            for i in range(22, len(d) - 14, 22):               # flechitas intermedias
                self._flecha(d[i], math.atan2(d[i + 3][1] - d[i - 3][1], d[i + 3][0] - d[i - 3][0]),
                             col, L=9, relleno="white")
            j = max(0, len(d) - 6)                             # flecha grande al final
            self._flecha(d[-1], math.atan2(d[-1][1] - d[j][1], d[-1][0] - d[j][0]), col, L=11)

        # 3) círculos numerados al inicio de cada trazo (se separan si se encimarían)
        usados = []
        for i, (t, col) in enumerate(zip(trazos, colores)):
            x, y = self._xy(t[0])
            if len(t) == 1:
                x, y = x + 11, y - 11
            for dx, dy in ((0, 0), (14, -14), (-14, -14), (14, 14), (-14, 14)):
                if all(math.hypot(x + dx - ux, y + dy - uy) > 17 for ux, uy in usados):
                    x, y = x + dx, y + dy
                    break
            usados.append((x, y))
            c.create_oval(x - 9, y - 9, x + 9, y + 9, fill=col, outline="white", width=2, tags="num")
            c.create_text(x, y, text=str(i + 1), font=("Arial", 10, "bold"), fill="white", tags="num")

    def descripcion(self, letra):
        trazos = TRAZOS_GUIA.get(letra, [])
        n = len(trazos)
        if n == 0:
            return "Esta letra no tiene guía."
        if n == 1:
            return "1 trazo continuo.\nNo levantes el lápiz."
        partes = [f"{n} trazos: sigue el orden 1 → {n}."]
        puntos = [str(i + 1) for i, t in enumerate(trazos) if len(t) == 1]
        if puntos:
            partes.append(f"El {' y '.join(puntos)} es solo un toque (punto).")
        partes.append("Levanta el lápiz en la línea punteada.")
        return "\n".join(partes)


class GuiaMixin:
    """Selector de letra + guía estática, común a Window2D y Window3D."""
    GUIA_COLOR = SUCCESS
    POR_FILA = 9

    def _build_selector(self, parent):
        ctk.CTkLabel(parent, text="Elige la letra", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=MUTED).pack(pady=(8, 2))
        marco = ctk.CTkFrame(parent, fg_color="transparent")
        marco.pack(padx=8, pady=(0, 10))
        self._btn_letras = {}
        for i, letra in enumerate(TRAZOS_GUIA):
            b = ctk.CTkButton(marco, text=letra, width=28, height=28, corner_radius=8,
                              fg_color=TILE, text_color=TEXT, hover_color="#dcdff5",
                              font=ctk.CTkFont(size=14, weight="bold"),
                              command=lambda l=letra: self.elegir_letra(l))
            b.grid(row=i // self.POR_FILA, column=i % self.POR_FILA, padx=1, pady=2)
            self._btn_letras[letra] = b
        self.elegir_letra(next(iter(TRAZOS_GUIA)))

    def elegir_letra(self, letra):
        self.letra_actual = letra
        for l, b in self._btn_letras.items():
            b.configure(fg_color=self.GUIA_COLOR if l == letra else TILE,
                        text_color="white" if l == letra else TEXT)
        self.guia_titulo.configure(text=f"Así se traza la «{letra}»")
        self.guia_anim.mostrar(letra)
        self.guia_texto.configure(text=self.guia_anim.descripcion(letra))


# ======================================================================
# Ventana A · Menú principal
# ======================================================================
class MainMenu(ctk.CTkFrame):
    def __init__(self, master, on_2d, on_3d):
        super().__init__(master, fg_color=BG)
        self.bg_canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.bg_canvas.bind("<Configure>", self._on_canvas_resize)
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.place(relx=0.5, rely=0.45, anchor="center")
        ctk.CTkLabel(main_container, text="UNIVERSIDAD DE LOS LLANOS", text_color=ACCENT,
                     font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold")).pack()
        title_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        title_frame.pack(pady=(5, 0))
        ctk.CTkLabel(title_frame, text="Reconocedor ", text_color=TEXT,
                     font=ctk.CTkFont(family="Segoe UI", size=46, weight="bold")).pack(side="left")
        ctk.CTkLabel(title_frame, text="HMM", text_color=ACCENT,
                     font=ctk.CTkFont(family="Segoe UI", size=46, weight="bold")).pack(side="left")
        ctk.CTkLabel(main_container, text="Trazos bidimensionales y esféricos", text_color=MUTED,
                     font=ctk.CTkFont(family="Segoe UI", size=18)).pack(pady=(0, 40))
        cards_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        cards_frame.pack()
        self._card_2d = self._create_card(
            parent=cards_frame, title="Modo 2D", subtitle="Plano cartesiano",
            btn_text="Ingresar al Lienzo", cmd=on_2d,
            theme_color=ACCENT, bg_color="#f4f7ff", is_active=True
        )
        self._card_2d.pack(side="left", padx=20)
        self._card_3d = self._create_card(
            parent=cards_frame, title="Modo 3D", subtitle="Superficie esférica",
            btn_text="Ingresar al Lienzo", cmd=on_3d,
            theme_color=SUCCESS, bg_color="#f2fdf6", is_active=True
        )
        self._card_3d.pack(side="left", padx=20)
        ctk.CTkLabel(self, text="Procesos Estocásticos · 603504 · 5.º Semestre", text_color=MUTED,
                     font=ctk.CTkFont(family="Segoe UI", size=12)).pack(side="bottom", pady=20)
    def _on_canvas_resize(self, event): self._draw_background_pattern(event.width, event.height)
    def _draw_background_pattern(self, w, h):
        self.bg_canvas.delete("all")
        letras = ["a", "b", "c", "x", "y", "z", "π", "λ", "α", "β", "∑", "∫"]
        colores_fondo = ["#cbd5e1", "#bfdbfe", "#bbf7d0", "#fecaca"]
        spacing = 160
        idx = 0
        for x in range(40, w + 100, spacing):
            for y in range(40, h + 100, spacing):
                char = letras[idx % len(letras)]
                color = colores_fondo[idx % len(colores_fondo)]
                size = 50 + (idx % 4) * 15
                offset_x = (spacing // 2) if (y // spacing) % 2 == 0 else 0
                self.bg_canvas.create_text(x + offset_x, y, text=char, fill=color, font=("Segoe UI", size, "bold"))
                idx += 1
        pts_top = []
        for x in range(0, w + 20, 20):
            y = 100 + math.sin(x / 150) * 40
            pts_top.extend([x, y])
        self.bg_canvas.create_line(*pts_top, fill="#cbd5e1", width=3, smooth=True)
        pts_bot = []
        for x in range(0, w + 20, 20):
            y = (h - 100) + math.cos(x / 120) * 40
            pts_bot.extend([x, y])
        self.bg_canvas.create_line(*pts_bot, fill="#cbd5e1", width=3, smooth=True)
    def _create_card(self, parent, title, subtitle, btn_text, cmd, theme_color, bg_color, is_active):
        shadow = ctk.CTkFrame(parent, fg_color="#e2e8f0", corner_radius=28, width=324, height=454)
        shadow.pack_propagate(False)
        card = ctk.CTkFrame(shadow, fg_color=bg_color, corner_radius=26, width=320, height=450, border_width=2, border_color=bg_color)
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)
        def on_enter(e):
            card.configure(border_color=theme_color)
            btn.configure(fg_color=theme_color if is_active else DANGER)
            btn.configure(text_color="white")
            canvas.itemconfigure("trazo", width=6)
        def on_leave(e):
            card.configure(border_color=bg_color)
            if is_active: btn.configure(fg_color=theme_color, text_color="white")
            else: btn.configure(fg_color="#fee2e2", text_color=DANGER)
            canvas.itemconfigure("trazo", width=4)
        for widget in (card, shadow):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
        canvas = tk.Canvas(card, width=240, height=200, bg=bg_color, highlightthickness=0)
        canvas.pack(pady=(30, 10))
        if title == "Modo 2D":
            canvas.create_line(30, 180, 210, 180, fill="#bfdbfe", width=3)
            canvas.create_line(30, 20, 30, 180, fill="#bfdbfe", width=3)
            pts = [(50, 150), (70, 110), (100, 80), (130, 100), (160, 60), (190, 70)]
            for i in range(len(pts) - 1):
                canvas.create_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], fill="#bfdbfe", width=2, dash=(2, 4))
            canvas.create_line(*[c for p in pts for c in p], fill=theme_color, width=4, smooth=True, capstyle="round", tags="trazo")
            canvas.create_oval(185, 65, 195, 75, fill=DANGER, outline="white", width=2)
        else:
            canvas.create_oval(60, 40, 180, 160, outline="#bbf7d0", width=3, fill="#dcfce7")
            canvas.create_arc(60, 60, 180, 140, start=0, extent=180, style="arc", outline="#86efac", width=4)
            canvas.create_arc(60, 60, 180, 140, start=180, extent=180, style="arc", outline="#86efac", width=4, dash=(4, 4))
            canvas.create_line(90, 50, 120, 110, 150, 70, fill=theme_color, width=4, smooth=True, capstyle="round", tags="trazo")
            canvas.create_oval(145, 65, 155, 75, fill=DANGER, outline="white", width=2)
        ctk.CTkLabel(card, text=title, text_color=TEXT, font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold")).pack(pady=(10, 2))
        ctk.CTkLabel(card, text=subtitle, text_color=MUTED, font=ctk.CTkFont(family="Segoe UI", size=14)).pack()
        btn_color = theme_color if is_active else "#fee2e2"
        txt_color = "white" if is_active else DANGER
        hover_color = "#1d4ed8" if is_active else DANGER
        btn = ctk.CTkButton(card, text=btn_text, width=220, height=48, corner_radius=24, fg_color=btn_color, hover_color=hover_color, text_color=txt_color, font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"), command=cmd)
        btn.pack(side="bottom", pady=30)
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return shadow

# ======================================================================
# Ventana C · Modo 3D
# ======================================================================
class Window3D(GuiaMixin, ctk.CTkFrame):
    GUIA_COLOR = SUCCESS
    ZOOM_MIN, ZOOM_MAX = 0.5, 2.5
    def __init__(self, master, recognizer, on_back):
        super().__init__(master, fg_color=BG)
        self.rec, self.on_back = recognizer, on_back
        self.size = 500
        self.offset = (0, 0)
        self._last_dim = None
        self.R = np.eye(3)
        self.zoom = 1.0
        self.strokes = []
        self.items = []
        self.drawing = False
        self._rotating = False
        self._rot_last = (0, 0)
        self._pending = None
        self._last_key = None
        self.proj_item = None
        self.icons, self.slots = {}, []
        self._grid = self._build_grid()
        self.letra_actual = None
        self.guia_anim = None
        self._build()
        self.reset_panel()
        top = self.winfo_toplevel()
        top.bind("<Control-z>", lambda e: self.undo() if self.winfo_ismapped() else None)
        top.bind("<Delete>", lambda e: self.clear_canvas() if self.winfo_ismapped() else None)

    @staticmethod
    def _make_icon_image(letter, px=128):
        S = 4; n = px * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, n - 1, n - 1), radius=int(n * 0.24), fill=TILE)
        font = None
        for name in ("segoeuib.ttf", "arialbd.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc", "LiberationSans-Bold.ttf"):
            try: font = ImageFont.truetype(name, int(n * 0.62)); break
            except OSError: continue
        if font is None:
            try: font = ImageFont.load_default(size=int(n * 0.62))
            except TypeError: font = ImageFont.load_default()
        bb = d.textbbox((0, 0), letter, font=font)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((n - w) / 2 - bb[0], (n - h) / 2 - bb[1]), letter, font=font, fill=SUCCESS)
        return img.resize((px, px), Image.LANCZOS)

    def _icon(self, letter):
        if letter not in self.icons:
            path = os.path.join(ICON_DIR, f"{letter}.png")
            if letter == "_vacio": img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
            elif os.path.exists(path): img = Image.open(path)
            else: img = self._make_icon_image(letter)
            self.icons[letter] = ctk.CTkImage(light_image=img, dark_image=img, size=(40, 40))
        return self.icons[letter]

    def _build(self):
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=320)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=24, pady=(18, 10))
        ctk.CTkButton(header, text="←  Menú", width=90, height=34, corner_radius=17, fg_color="#6366f1", text_color="white", hover_color="#4f46e5", command=self.on_back).pack(side="left")
        ctk.CTkLabel(header, text="Reconocedor HMM", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT).pack(side="left", padx=(16, 8))
        ctk.CTkLabel(header, text="Modo 3D", font=ctk.CTkFont(size=13, weight="bold"), text_color="white", fg_color=SUCCESS, corner_radius=10, width=70, height=26).pack(side="left")

        col_guia = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22)
        col_guia.grid(row=1, column=0, sticky="nsew", padx=(24, 12), pady=(0, 24))
        ctk.CTkLabel(col_guia, text="Guía de trazo", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(pady=(20, 10))
        self.guia_frame = ctk.CTkFrame(col_guia, fg_color="#f8fafc", corner_radius=15, border_width=2, border_color="#e2e8f0")
        self.guia_frame.pack(fill="x", padx=15, pady=10)
        self.guia_titulo = ctk.CTkLabel(self.guia_frame, text="Así se traza", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT)
        self.guia_titulo.pack(pady=(15, 5))
        self.guia_canvas = tk.Canvas(self.guia_frame, bg="#f8fafc", width=240, height=200, highlightthickness=0)
        self.guia_canvas.pack(pady=5)
        self.guia_texto = ctk.CTkLabel(self.guia_frame, text="", font=ctk.CTkFont(size=13), text_color=MUTED, wraplength=240)
        self.guia_texto.pack(pady=(0, 15))
        self.guia_anim = GuiaEstatica(self.guia_canvas)
        self._build_selector(col_guia)

        col_lienzo = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22)
        col_lienzo.grid(row=1, column=1, sticky="nsew", padx=12, pady=(0, 24))
        tools = ctk.CTkFrame(col_lienzo, fg_color="transparent")
        tools.pack(fill="x", padx=18, pady=(16, 10))
        ctk.CTkButton(tools, text="↶  Deshacer", width=110, height=36, corner_radius=18, fg_color=TILE, text_color=TEXT, hover_color="#dcdff5", command=self.undo).pack(side="left", padx=(0, 8))
        ctk.CTkButton(tools, text="Limpiar lienzo", width=130, height=36, corner_radius=18, fg_color=DANGER, hover_color="#c93a3f", text_color="white", command=self.clear_canvas).pack(side="left", padx=(0, 8))
        ctk.CTkButton(tools, text="⟳  Vista", width=90, height=36, corner_radius=18, fg_color=TILE, text_color=TEXT, hover_color="#dcdff5", command=self.reset_view).pack(side="left", padx=(0, 8))
        self.canvas = tk.Canvas(col_lienzo, bg=CARD, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=18, pady=(0, 8))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.ring = self.canvas.create_oval(0, 0, 0, 0, outline=SUCCESS, width=2, state="hidden", tags="ring")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", lambda e: self._move_ring(e.x, e.y))
        self.canvas.bind("<Leave>", lambda e: self.canvas.itemconfigure(self.ring, state="hidden"))
        for b in (2, 3):
            self.canvas.bind(f"<ButtonPress-{b}>", self._start_rot)
            self.canvas.bind(f"<B{b}-Motion>", self._rot_motion)
            self.canvas.bind(f"<ButtonRelease-{b}>", self._end_rot)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)
        self.lbl_status = ctk.CTkLabel(col_lienzo, text="Clic izq.: dibujar  ·  Clic der.: rotar  ·  Rueda: zoom", font=ctk.CTkFont(size=13), text_color=MUTED)
        self.lbl_status.pack(pady=(0, 14))

        right = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22)
        right.grid(row=1, column=2, sticky="nsew", padx=(12, 24), pady=(0, 24))
        right.grid_columnconfigure(0, weight=1)
        hero = ctk.CTkFrame(right, fg_color=SUCCESS, corner_radius=18)
        hero.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        self.lbl_hero = ctk.CTkLabel(hero, text="?", width=88, height=88, corner_radius=16, fg_color="white", text_color=SUCCESS, font=ctk.CTkFont(size=50, weight="bold"))
        self.lbl_hero.pack(side="left", padx=16, pady=16)
        info = ctk.CTkFrame(hero, fg_color="transparent")
        info.pack(side="left", padx=6)
        ctk.CTkLabel(info, text="Letra más probable", text_color="#dcdcff", font=ctk.CTkFont(size=12)).pack(anchor="w")
        self.lbl_hero_p = ctk.CTkLabel(info, text="—", text_color="white", width=150, anchor="w", font=ctk.CTkFont(size=32, weight="bold"))
        self.lbl_hero_p.pack(anchor="w")
        ctk.CTkLabel(right, text=f"Top {TOP_N} · P(carácter | O₁:t)", text_color=MUTED, font=ctk.CTkFont(size=12, weight="bold")).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 4))
        lst = ctk.CTkFrame(right, fg_color="transparent")
        lst.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 16))
        lst.grid_columnconfigure(0, weight=1)
        for i in range(TOP_N):
            row = ctk.CTkFrame(lst, fg_color=ROW_FIRST if i == 0 else "transparent", corner_radius=14)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(3, weight=1)
            rank = ctk.CTkLabel(row, text=str(i + 1), width=22, text_color=SUCCESS if i == 0 else MUTED, font=ctk.CTkFont(size=13, weight="bold"))
            rank.grid(row=0, column=0, padx=(10, 2), pady=5)
            icon = ctk.CTkLabel(row, text="", image=self._icon("_vacio"), width=40, height=40)
            icon.grid(row=0, column=1, padx=(2, 8))
            name = ctk.CTkLabel(row, text="·", width=26, text_color=SUCCESS if i == 0 else TEXT, font=ctk.CTkFont(size=22, weight="bold"))
            name.grid(row=0, column=2, padx=(0, 10))
            bar = ctk.CTkProgressBar(row, height=10, corner_radius=5, fg_color="#e9ebf6", progress_color=SUCCESS if i == 0 else (ACCENT_SOFT if i < 3 else BAR_OFF))
            bar.grid(row=0, column=3, sticky="ew")
            bar.set(0)
            pct = ctk.CTkLabel(row, text="—", width=64, anchor="e", text_color=TEXT, font=ctk.CTkFont(size=14, weight="bold" if i == 0 else "normal"))
            pct.grid(row=0, column=4, padx=(10, 12))
            self.slots.append({"letter": None, "icon": icon, "name": name, "bar": bar, "pct": pct})

    @staticmethod
    def _build_grid():
        lines = []
        lat = np.radians(np.linspace(-90, 90, 37))
        for lon_deg in range(0, 360, 30):
            lon = np.radians(lon_deg)
            lines.append(np.column_stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)]))
        lon = np.radians(np.linspace(0, 360, 73))
        for lat_deg in (-60, -30, 0, 30, 60):
            la = np.radians(lat_deg)
            lines.append(np.column_stack([np.cos(la) * np.sin(lon), np.full_like(lon, np.sin(la)), np.cos(la) * np.cos(lon)]))
        return lines

    def _geom(self):
        ox, oy = self.offset
        cx = self.size / 2 + ox
        cy = self.size / 2 + oy
        rr = (self.size / 2 - 20) * self.zoom
        return cx, cy, rr

    def _cam_to_screen(self, p):
        cx, cy, rr = self._geom()
        return cx + p[0] * rr, cy - p[1] * rr

    def _map_to_sphere(self, mx, my):
        cx, cy, rr = self._geom()
        nx = (mx - cx) / rr
        ny = -(my - cy) / rr
        d2 = nx * nx + ny * ny
        if d2 > 1.0:
            d = math.sqrt(d2)
            nx /= d
            ny /= d
            nz = 0.0
        else:
            nz = math.sqrt(1.0 - d2)
        return np.array([nx, ny, nz])

    def _visible_runs(self, cam):
        runs, cur = [], []
        for p in cam:
            if p[2] >= -1e-6: cur.append(self._cam_to_screen(p))
            elif cur:
                runs.append(cur)
                cur = []
        if cur: runs.append(cur)
        return runs

    def _draw_stroke(self, idx):
        pts = np.asarray(self.strokes[idx], dtype=float)
        cam = pts @ self.R.T
        ids = []
        for run in self._visible_runs(cam):
            if len(run) >= 2:
                flat = [c for q in run for c in q]
                ids.append(self.canvas.create_line(*flat, fill=INK, width=BRUSH, smooth=True, splinesteps=8, capstyle="round", joinstyle="round", tags="ink"))
            else:
                x, y = run[0]
                r = BRUSH / 2.0
                ids.append(self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=INK, outline=INK, tags="ink"))
        return ids

    def _redraw_scene(self):
        c = self.canvas
        c.delete("surface")
        c.delete("ink")
        cx, cy, rr = self._geom()
        c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=SURFACE, outline=SURFACE_EDGE, width=2, tags="surface")
        for line in self._grid:
            for run in self._visible_runs(line @ self.R.T):
                if len(run) >= 2:
                    c.create_line(*[v for q in run for v in q], fill="#d5dde8", width=1, tags="surface")
        self.items = [self._draw_stroke(i) for i in range(len(self.strokes))]
        c.tag_lower("surface")
        c.tag_raise("proj")
        c.tag_raise("ring")

    def _on_canvas_resize(self, event):
        dim = (event.width, event.height)
        if dim == self._last_dim: return
        self._last_dim = dim
        self.size = max(200, min(event.width, event.height))
        self.offset = ((event.width - self.size) / 2, (event.height - self.size) / 2)
        self._redraw_scene()
        self._show_hint()

    def _show_hint(self):
        self.canvas.delete("hint")
        if not self.strokes:
            ox, oy = self.offset
            x = ox + self.size / 2
            y = oy + self.size / 2
            self.canvas.create_text(x, y, text="Dibuja aquí sobre la esfera", fill="#b3b8d1", font=("Helvetica", 15), tags="hint")

    def _move_ring(self, x, y):
        r = BRUSH + 1
        self.canvas.coords(self.ring, x - r, y - r, x + r, y + r)
        self.canvas.itemconfigure(self.ring, state="normal")
        self.canvas.tag_raise("ring")

    def _start_rot(self, e): self._rotating = True; self._rot_last = (e.x, e.y)
    def _end_rot(self, e): self._rotating = False
    def _rot_motion(self, e):
        self._move_ring(e.x, e.y)
        if self._rotating: self._do_rot(e)

    def _do_rot(self, e):
        lx, ly = self._rot_last
        dx, dy = e.x - lx, e.y - ly
        self._rot_last = (e.x, e.y)
        if dx == 0 and dy == 0: return
        _, _, rr = self._geom()
        a, b = dx / rr, dy / rr
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)
        Ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        Rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
        R = Rx @ Ry @ self.R
        u, _, vt = np.linalg.svd(R)
        self.R = u @ vt
        self._hide_projection()
        self._redraw_scene()
        self.schedule_update()

    def on_wheel(self, e):
        up = getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0
        f = 1.1 if up else 1 / 1.1
        self.zoom = min(self.ZOOM_MAX, max(self.ZOOM_MIN, self.zoom * f))
        self._hide_projection()
        self._redraw_scene()
        self.schedule_update()

    def reset_view(self):
        self.R = np.eye(3)
        self.zoom = 1.0
        self._hide_projection()
        self._redraw_scene()
        self.schedule_update()

    def on_press(self, e):
        if e.state & 0x0001: self._start_rot(e); return
        self.drawing = True
        self.canvas.delete("hint")
        p_obj = self.R.T @ self._map_to_sphere(e.x, e.y)
        self.strokes.append([p_obj])
        self.items.append(self._draw_stroke(len(self.strokes) - 1))
        self._move_ring(e.x, e.y)
        self.schedule_update()

    def on_move(self, e):
        self._move_ring(e.x, e.y)
        if self._rotating: self._do_rot(e); return
        if not self.drawing or not self.strokes: return
        _, _, rr = self._geom()
        p_cam = self._map_to_sphere(e.x, e.y)
        stroke = self.strokes[-1]
        last_cam = self.R @ stroke[-1]
        if np.linalg.norm(p_cam - last_cam) * rr < MIN_STEP: return
        stroke.append(self.R.T @ p_cam)
        for i in self.items[-1]: self.canvas.delete(i)
        self.items[-1] = self._draw_stroke(len(self.strokes) - 1)
        self.canvas.tag_raise("proj")
        self.canvas.tag_raise("ring")
        self.schedule_update()

    def on_release(self, e):
        if self._rotating and not self.drawing:
            self._rotating = False
            return
        self.drawing = False
        self.do_update()

    def _data_strokes(self):
        return [[tuple((self.R @ p) * 100.0) for p in s] for s in self.strokes]

    def _ranking(self, data):
        seq = self.rec.extractor.extract(data)
        if not seq: return []
        models = self.rec.classifier.models
        chars = list(models.keys())
        ll = np.array([models[c].log_likelihood(seq) for c in chars]) / TEMPERATURE
        ll -= ll.max()
        p = np.exp(ll)
        p /= p.sum()
        order = np.argsort(-p)
        return [(chars[i], float(p[i])) for i in order]

    def schedule_update(self):
        if self._pending is None: self._pending = self.after(UPDATE_MS, self.do_update)

    def do_update(self):
        self._pending = None
        if not self.strokes: return
        data = self._data_strokes()
        ranking = self._ranking(data)
        if not ranking:
            self._hide_projection()
            return
        top = [(l, round(p, 3)) for l, p in ranking[:TOP_N]]
        if top != self._last_key:
            self._last_key = top
            self._show_top(ranking[:TOP_N])
        self._draw_projection(data)

    def _show_top(self, top):
        for slot, (letter, p) in zip(self.slots, top):
            if slot["letter"] != letter:
                slot["letter"] = letter
                slot["icon"].configure(image=self._icon(letter))
                slot["name"].configure(text=letter)
            slot["bar"].set(float(p))
            slot["pct"].configure(text=f"{100 * p:.1f}%")
        best, pb = top[0]
        self.lbl_hero.configure(text=best)
        self.lbl_hero_p.configure(text=f"{100 * pb:.1f}%")
        self.lbl_status.configure(text=f"Reconociendo como «{best}»")

    def _hide_projection(self):
        if self.proj_item is not None: self.canvas.itemconfigure(self.proj_item, state="hidden")

    def _draw_projection(self, data):
        res = self.rec.autocomplete(data)
        if not res or not res[1]:
            self._hide_projection()
            return
        last = np.asarray(data[-1][-1], dtype=float)
        nrm = np.linalg.norm(last)
        if nrm == 0:
            self._hide_projection()
            return
        last = last / nrm
        if last[2] < 0:
            self._hide_projection()
            return
        path = list(self._cam_to_screen(last))
        for p in res[1]:
            if p[2] < 0: break
            path += list(self._cam_to_screen(p))
        if len(path) < 4:
            self._hide_projection()
            return
        if self.proj_item is None:
            self.proj_item = self.canvas.create_line(*path, fill="#86efac", width=4, dash=(2, 7), capstyle="round", smooth=True, tags="proj")
        else:
            self.canvas.coords(self.proj_item, *path)
            self.canvas.itemconfigure(self.proj_item, state="normal")
        self.canvas.tag_raise("ink")
        self.canvas.tag_raise("ring")

    def reset_panel(self):
        self._last_key = None
        for slot in self.slots:
            slot["letter"] = None
            slot["icon"].configure(image=self._icon("_vacio"))
            slot["name"].configure(text="·")
            slot["bar"].set(0)
            slot["pct"].configure(text="—")
        self.lbl_hero.configure(text="?")
        self.lbl_hero_p.configure(text="—")
        self.lbl_status.configure(text="Clic izq.: dibujar  ·  Clic der.: rotar  ·  Rueda: zoom")

    def undo(self):
        if not self.strokes: return
        self.strokes.pop()
        for i in self.items.pop(): self.canvas.delete(i)
        self.drawing = False
        if self.strokes:
            self._last_key = None
            self.do_update()
        else:
            self._hide_projection()
            self.reset_panel()
            self._show_hint()

    def clear_canvas(self):
        self.strokes, self.items = [], []
        self.drawing = False
        self.canvas.delete("ink")
        self._hide_projection()
        self.reset_panel()
        self._show_hint()

# ======================================================================
# Ventana B · Modo 2D
# ======================================================================
class Window2D(GuiaMixin, ctk.CTkFrame):
    GUIA_COLOR = ACCENT
    def __init__(self, master, recognizer, on_back):
        super().__init__(master, fg_color=BG)
        self.rec, self.on_back = recognizer, on_back
        self.size = 500
        self.S = DATA_SIZE / self.size
        self.offset = (0, 0)
        self.strokes = []
        self.items = []
        self.disp = []
        self.cur_kind = None
        self.drawing = False
        self._pending = None
        self._last_key = None
        self.proj_item = None
        self.icons, self.slots = {}, []
        self.letra_actual = None
        self.guia_anim = None
        self._build()
        self.reset_panel()
        top = self.winfo_toplevel()
        top.bind("<Control-z>", lambda e: self.undo() if self.winfo_ismapped() else None)
        top.bind("<Delete>", lambda e: self.clear_canvas() if self.winfo_ismapped() else None)

    @staticmethod
    def _make_icon_image(letter, px=128):
        S = 4; n = px * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, n - 1, n - 1), radius=int(n * 0.24), fill=TILE)
        font = None
        for name in ("segoeuib.ttf", "arialbd.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc", "LiberationSans-Bold.ttf"):
            try: font = ImageFont.truetype(name, int(n * 0.62)); break
            except OSError: continue
        if font is None:
            try: font = ImageFont.load_default(size=int(n * 0.62))
            except TypeError: font = ImageFont.load_default()
        bb = d.textbbox((0, 0), letter, font=font)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((n - w) / 2 - bb[0], (n - h) / 2 - bb[1]), letter, font=font, fill=ACCENT)
        return img.resize((px, px), Image.LANCZOS)

    def _icon(self, letter):
        if letter not in self.icons:
            path = os.path.join(ICON_DIR, f"{letter}.png")
            if letter == "_vacio": img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
            elif os.path.exists(path): img = Image.open(path)
            else: img = self._make_icon_image(letter)
            self.icons[letter] = ctk.CTkImage(light_image=img, dark_image=img, size=(40, 40))
        return self.icons[letter]

    def _build(self):
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=320)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=3, sticky="ew", padx=24, pady=(18, 10))
        ctk.CTkButton(header, text="←  Menú", width=90, height=34, corner_radius=17, fg_color="#6366f1", text_color="white", hover_color="#4f46e5", command=self.on_back).pack(side="left")
        ctk.CTkLabel(header, text="Reconocedor HMM", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT).pack(side="left", padx=(16, 8))
        ctk.CTkLabel(header, text="Modo 2D", font=ctk.CTkFont(size=13, weight="bold"), text_color="white", fg_color=ACCENT, corner_radius=10, width=70, height=26).pack(side="left")

        col_guia = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22)
        col_guia.grid(row=1, column=0, sticky="nsew", padx=(24, 12), pady=(0, 24))
        ctk.CTkLabel(col_guia, text="Guía de trazo", font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(pady=(20, 10))
        self.guia_frame = ctk.CTkFrame(col_guia, fg_color="#f8fafc", corner_radius=15, border_width=2, border_color="#e2e8f0")
        self.guia_frame.pack(fill="x", padx=15, pady=10)
        self.guia_titulo = ctk.CTkLabel(self.guia_frame, text="Así se traza", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT)
        self.guia_titulo.pack(pady=(15, 5))
        self.guia_canvas = tk.Canvas(self.guia_frame, bg="#f8fafc", width=240, height=200, highlightthickness=0)
        self.guia_canvas.pack(pady=5)
        self.guia_texto = ctk.CTkLabel(self.guia_frame, text="", font=ctk.CTkFont(size=13), text_color=MUTED, wraplength=240)
        self.guia_texto.pack(pady=(0, 15))
        self.guia_anim = GuiaEstatica(self.guia_canvas)
        self._build_selector(col_guia)

        col_lienzo = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22)
        col_lienzo.grid(row=1, column=1, sticky="nsew", padx=12, pady=(0, 24))
        tools = ctk.CTkFrame(col_lienzo, fg_color="transparent")
        tools.pack(fill="x", padx=18, pady=(16, 10))
        ctk.CTkButton(tools, text="↶  Deshacer", width=110, height=36, corner_radius=18, fg_color=TILE, text_color=TEXT, hover_color="#dcdff5", command=self.undo).pack(side="left", padx=(0, 8))
        ctk.CTkButton(tools, text="Limpiar lienzo", width=130, height=36, corner_radius=18, fg_color=DANGER, hover_color="#c93a3f", text_color="white", command=self.clear_canvas).pack(side="left", padx=(0, 8))
        
        self.canvas = tk.Canvas(col_lienzo, bg=CARD, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=18, pady=(0, 8))
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.ring = self.canvas.create_oval(0, 0, 0, 0, outline=ACCENT, width=2, state="hidden", tags="ring")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", lambda e: self._move_ring(e.x, e.y))
        self.canvas.bind("<Leave>", lambda e: self.canvas.itemconfigure(self.ring, state="hidden"))

        self.lbl_status = ctk.CTkLabel(col_lienzo, text="", font=ctk.CTkFont(size=13), text_color=MUTED)
        self.lbl_status.pack(pady=(0, 14))

        right = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22)
        right.grid(row=1, column=2, sticky="nsew", padx=(12, 24), pady=(0, 24))
        right.grid_columnconfigure(0, weight=1)
        hero = ctk.CTkFrame(right, fg_color=ACCENT, corner_radius=18)
        hero.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        self.lbl_hero = ctk.CTkLabel(hero, text="?", width=88, height=88, corner_radius=16, fg_color="white", text_color=ACCENT, font=ctk.CTkFont(size=50, weight="bold"))
        self.lbl_hero.pack(side="left", padx=16, pady=16)
        info = ctk.CTkFrame(hero, fg_color="transparent")
        info.pack(side="left", padx=6)
        ctk.CTkLabel(info, text="Letra más probable", text_color="#dcdcff", font=ctk.CTkFont(size=12)).pack(anchor="w")
        self.lbl_hero_p = ctk.CTkLabel(info, text="—", text_color="white", width=150, anchor="w", font=ctk.CTkFont(size=32, weight="bold"))
        self.lbl_hero_p.pack(anchor="w")
        ctk.CTkLabel(right, text=f"Top {TOP_N} · P(carácter | O₁:t)", text_color=MUTED, font=ctk.CTkFont(size=12, weight="bold")).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 4))
        lst = ctk.CTkFrame(right, fg_color="transparent")
        lst.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 16))
        lst.grid_columnconfigure(0, weight=1)
        for i in range(TOP_N):
            row = ctk.CTkFrame(lst, fg_color=ROW_FIRST if i == 0 else "transparent", corner_radius=14)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(3, weight=1)
            rank = ctk.CTkLabel(row, text=str(i + 1), width=22, text_color=ACCENT if i == 0 else MUTED, font=ctk.CTkFont(size=13, weight="bold"))
            rank.grid(row=0, column=0, padx=(10, 2), pady=5)
            icon = ctk.CTkLabel(row, text="", image=self._icon("_vacio"), width=40, height=40)
            icon.grid(row=0, column=1, padx=(2, 8))
            name = ctk.CTkLabel(row, text="·", width=26, text_color=ACCENT if i == 0 else TEXT, font=ctk.CTkFont(size=22, weight="bold"))
            name.grid(row=0, column=2, padx=(0, 10))
            bar = ctk.CTkProgressBar(row, height=10, corner_radius=5, fg_color="#e9ebf6", progress_color=ACCENT if i == 0 else (ACCENT_SOFT if i < 3 else BAR_OFF))
            bar.grid(row=0, column=3, sticky="ew")
            bar.set(0)
            pct = ctk.CTkLabel(row, text="—", width=64, anchor="e", text_color=TEXT, font=ctk.CTkFont(size=14, weight="bold" if i == 0 else "normal"))
            pct.grid(row=0, column=4, padx=(10, 12))
            self.slots.append({"letter": None, "icon": icon, "name": name, "bar": bar, "pct": pct})

    def _on_canvas_resize(self, event):
        self.canvas.delete("surface")
        self.size = max(200, min(event.width, event.height))
        self.S = DATA_SIZE / self.size
        c, s, r = self.canvas, self.size, 22
        x1 = y1 = 2
        x2 = y2 = s - 2
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        self.offset = ((event.width - s) / 2, (event.height - s) / 2)
        ox, oy = self.offset
        pts_shifted = [p + (ox if i % 2 == 0 else oy) for i, p in enumerate(pts)]
        c.create_polygon(pts_shifted, smooth=True, fill=SURFACE, outline=SURFACE_EDGE, width=2, tags="surface")
        
        self.canvas.create_line(ox + s*0.1, oy + s*0.2, ox + s*0.9, oy + s*0.2, fill="#e2e8f0", width=1, dash=(2, 4), tags="surface")
        self.canvas.create_line(ox + s*0.1, oy + s*0.5, ox + s*0.9, oy + s*0.5, fill="#e2e8f0", width=1, dash=(2, 4), tags="surface")
        self.canvas.create_line(ox + s*0.1, oy + s*0.8, ox + s*0.9, oy + s*0.8, fill="#e2e8f0", width=1, dash=(2, 4), tags="surface")
        
        self._show_hint()
        self._redraw_all_strokes()

    def _redraw_all_strokes(self):
        self.canvas.delete("ink")
        self.items = []
        ox, oy = self.offset
        for stroke in self.strokes:
            disp_stroke = [(x + ox, y + oy) for x, y in stroke]
            flat = [c for p in disp_stroke for c in p]
            if len(flat) >= 4:
                item = self.canvas.create_line(*flat, fill=INK, width=BRUSH, smooth=True, splinesteps=8, capstyle="round", joinstyle="round", tags="ink")
                self.items.append(item)
            elif len(flat) == 2:
                x, y = flat
                r = BRUSH / 2.0
                item = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=INK, outline=INK, tags="ink")
                self.items.append(item)
        self.canvas.tag_raise("ring")

    def _show_hint(self):
        self.canvas.delete("hint")
        if not self.strokes:
            ox, oy = self.offset
            x = ox + self.size / 2
            y = oy + self.size / 2
            self.canvas.create_text(x, y, text="Dibuja aquí una letra minúscula", fill="#b3b8d1", font=("Helvetica", 15), tags="hint")

    def _move_ring(self, x, y):
        r = BRUSH + 1
        self.canvas.coords(self.ring, x - r, y - r, x + r, y + r)
        self.canvas.itemconfigure(self.ring, state="normal")
        self.canvas.tag_raise("ring")

    def on_press(self, e):
        self.drawing = True
        self.canvas.delete("hint")
        self.strokes.append([(e.x, e.y)])
        self.disp = [(e.x, e.y)]
        r = BRUSH / 2.0
        item = self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill=INK, outline=INK, tags="ink")
        self.items.append(item)
        self.cur_kind = "dot"
        self._move_ring(e.x, e.y)
        self.schedule_update()

    def on_move(self, e):
        self._move_ring(e.x, e.y)
        if not self.drawing or not self.strokes: return
        stroke = self.strokes[-1]
        lx, ly = stroke[-1]
        if (e.x - lx) ** 2 + (e.y - ly) ** 2 < MIN_STEP ** 2: return
        stroke.append((e.x, e.y))
        px, py = self.disp[-1]
        self.disp.append((px + (e.x - px) * FOLLOW, py + (e.y - py) * FOLLOW))
        self._update_line()
        self.schedule_update()

    def on_release(self, e):
        if self.drawing and self.strokes and self.cur_kind == "line":
            self.disp.append(self.strokes[-1][-1])
            self._update_line()
        self.drawing = False
        self.do_update()

    def _update_line(self):
        flat = [c for p in self.disp for c in p]
        if self.cur_kind == "dot":
            self.canvas.delete(self.items[-1])
            self.items[-1] = self.canvas.create_line(*flat, fill=INK, width=BRUSH, smooth=True, splinesteps=8, capstyle="round", joinstyle="round", tags="ink")
            self.cur_kind = "line"
        else:
            self.canvas.coords(self.items[-1], *flat)
        self.canvas.tag_raise("ring")

    def _data_strokes(self):
        ox, oy = self.offset
        return [[((x - ox) * self.S, (y - oy) * self.S) for x, y in s] for s in self.strokes]

    def _ranking(self, data):
        seq = self.rec.extractor.extract(data)
        if not seq: return []
        models = self.rec.classifier.models
        chars = list(models.keys())
        ll = np.array([models[c].log_likelihood(seq) for c in chars]) / TEMPERATURE
        ll -= ll.max()
        p = np.exp(ll)
        p /= p.sum()
        order = np.argsort(-p)
        return [(chars[i], float(p[i])) for i in order]

    def schedule_update(self):
        if self._pending is None: self._pending = self.after(UPDATE_MS, self.do_update)

    def do_update(self):
        self._pending = None
        if not self.strokes: return
        data = self._data_strokes()
        ranking = self._ranking(data)
        if not ranking:
            self._hide_projection()
            return
        top = [(l, round(p, 3)) for l, p in ranking[:TOP_N]]
        if top != self._last_key:
            self._last_key = top
            self._show_top(ranking[:TOP_N])
        self._draw_projection(data)

    def _show_top(self, top):
        for slot, (letter, p) in zip(self.slots, top):
            if slot["letter"] != letter:
                slot["letter"] = letter
                slot["icon"].configure(image=self._icon(letter))
                slot["name"].configure(text=letter)
            slot["bar"].set(float(p))
            slot["pct"].configure(text=f"{100 * p:.1f}%")
        best, pb = top[0]
        self.lbl_hero.configure(text=best)
        self.lbl_hero_p.configure(text=f"{100 * pb:.1f}%")
        self.lbl_status.configure(text=f"Autocompletando como «{best}»")

    def _hide_projection(self):
        if self.proj_item is not None: self.canvas.itemconfigure(self.proj_item, state="hidden")

    def _draw_projection(self, data):
        res = self.rec.autocomplete(data)
        if not res or not res[1]:
            self._hide_projection()
            return
        ox, oy = self.offset
        x, y = self.strokes[-1][-1]
        path = [x, y]
        for px, py in res[1]:
            path += [(px / self.S) + ox, (py / self.S) + oy]
        if len(path) < 4:
            self._hide_projection()
            return
        if self.proj_item is None:
            self.proj_item = self.canvas.create_line(*path, fill=ACCENT_SOFT, width=4, dash=(2, 7), capstyle="round", smooth=True, tags="proj")
        else:
            self.canvas.coords(self.proj_item, *path)
            self.canvas.itemconfigure(self.proj_item, state="normal")
        self.canvas.tag_raise("ink")
        self.canvas.tag_raise("ring")

    def reset_panel(self):
        self._last_key = None
        for slot in self.slots:
            slot["letter"] = None
            slot["icon"].configure(image=self._icon("_vacio"))
            slot["name"].configure(text="·")
            slot["bar"].set(0)
            slot["pct"].configure(text="—")
        self.lbl_hero.configure(text="?")
        self.lbl_hero_p.configure(text="—")
        self.lbl_status.configure(text="Empieza a dibujar con el ratón o con el dedo")

    def undo(self):
        if not self.strokes: return
        self.strokes.pop()
        self.canvas.delete(self.items.pop())
        self.drawing = False
        if self.strokes:
            self._last_key = None
            self.do_update()
        else:
            self._hide_projection()
            self.reset_panel()
            self._show_hint()

    def clear_canvas(self):
        self.strokes, self.items, self.disp = [], [], []
        self.drawing = False
        self.canvas.delete("ink")
        self._hide_projection()
        self.reset_panel()
        self._show_hint()

# ======================================================================
# Aplicación principal
# ======================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("Reconocedor HMM · Procesos Estocásticos")
        w, h = 1250, 760
        x = max(0, (self.winfo_screenwidth() - w) // 2)
        y = max(0, (self.winfo_screenheight() - h) // 2 - 20)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1100, 660)
        self.configure(fg_color=BG)
        self.recognizer_2d = self._load_model_2d()
        self.recognizer_3d = self._load_model_3d()
        self.current = None
        self.window2d = None
        self.window3d = None
        self.menu = MainMenu(self, on_2d=self.open_2d, on_3d=self.open_3d)
        self.show(self.menu)

    @staticmethod
    def _load_model_2d():
        try: classifier = ModelRepository(MODEL_2D).load()
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el modelo:\n{MODEL_2D}\nEntrénalo con: python modelo_2d/train_2d.py")
            return None
        return Recognizer(Extractor2D(k_dir=16, min_dist=MIN_DIST_2D), classifier)

    @staticmethod
    def _load_model_3d():
        try: classifier = ModelRepository(MODEL_3D).load()
        except FileNotFoundError: return None
        return Recognizer(Extractor3D(k_dir=16, min_angle_dist=MIN_ANGLE_3D), classifier)

    def show(self, frame):
        if self.current is not None: self.current.pack_forget()
        self.current = frame
        frame.pack(fill="both", expand=True)

    def go_menu(self): self.show(self.menu)
    def open_2d(self):
        if self.recognizer_2d is None:
            self.recognizer_2d = self._load_model_2d()
            if self.recognizer_2d is None: return
        if self.window2d is None: self.window2d = Window2D(self, self.recognizer_2d, on_back=self.go_menu)
        self.show(self.window2d)
    def open_3d(self):
        if self.recognizer_3d is None:
            self.recognizer_3d = self._load_model_3d()
            if self.recognizer_3d is None:
                messagebox.showerror("Error", f"No se encontró el modelo 3D:\n{MODEL_3D}\n\nAsegúrate de haber ejecutado:\npython modelo_3d/train_3d.py")
                return
        if self.window3d is None: self.window3d = Window3D(self, self.recognizer_3d, on_back=self.go_menu)
        self.show(self.window3d)

if __name__ == "__main__":
    app = App()
    app.mainloop()