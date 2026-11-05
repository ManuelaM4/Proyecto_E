"""
GUI.py
Interfaz gráfica del modo 2D.
"""
import tkinter as tk
from tkinter import messagebox
from Recognizer import Recognizer
from FeatureExtractor import Extractor2D
from ModelRepository import ModelRepository
import os

class App2D:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor HMM 2D - Procesos Estocásticos")
        self.root.geometry("1000x650")
        self.root.configure(bg="#f0f0f0")

        self.extractor = Extractor2D(k_dir=16, min_dist=15.0)
        model_path = os.path.join("modelo_2d", "models_2d.json")
        
        try:
            repo = ModelRepository(model_path)
            classifier = repo.load()
            self.recognizer = Recognizer(self.extractor, classifier)
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el modelo en {model_path}.\nEjecuta train_2d.py primero.")
            self.root.destroy()
            return

        self.strokes = []
        self.current_stroke = []
        self.is_drawing = False

        self._build_ui()

    def _build_ui(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        left_frame = tk.Frame(main_frame, bg="white", bd=2, relief=tk.RIDGE)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(left_frame, bg="white", cursor="pencil")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        btn_frame = tk.Frame(left_frame, bg="white")
        btn_frame.pack(side=tk.BOTTOM, pady=10)
        tk.Button(btn_frame, text="Limpiar Lienzo", command=self.clear_canvas, bg="#e74c3c", fg="white", relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=5)

        right_frame = tk.Frame(main_frame, bg="#ffffff", width=300, bd=2, relief=tk.RIDGE)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="Clasificación en Tiempo Real", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
        self.prob_frame = tk.Frame(right_frame, bg="white")
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def start_draw(self, event):
        self.is_drawing = True
        self.current_stroke = [(event.x, event.y)]
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black")

    def draw(self, event):
        if self.is_drawing:
            x, y = event.x, event.y
            if self.current_stroke:
                px, py = self.current_stroke[-1]
                self.canvas.create_line(px, py, x, y, fill="black", width=2, smooth=True)
            self.current_stroke.append((x, y))
            self.update_predictions()

    def stop_draw(self, event):
        if self.is_drawing and len(self.current_stroke) > 1:
            self.strokes.append(self.current_stroke)
        self.is_drawing = False
        self.current_stroke = []
        self.update_predictions()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.strokes = []
        self.current_stroke = []
        self.update_predictions()

    def update_predictions(self):
        for widget in self.prob_frame.winfo_children():
            widget.destroy()

        all_strokes = self.strokes.copy()
        if self.current_stroke:
            all_strokes.append(self.current_stroke)

        if not all_strokes:
            return

        posteriors = self.recognizer.posteriors(all_strokes)
        if not posteriors:
            return

        for i, (char, prob) in enumerate(posteriors[:5]):
            if prob < 0.01: break
            f = tk.Frame(self.prob_frame, bg="#ecf0f1" if i == 0 else "white")
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=char.upper(), font=("Arial", 16, "bold"), bg=f["bg"], width=3).pack(side=tk.LEFT)
            tk.Label(f, text=f"{prob*100:.1f}%", font=("Arial", 12), bg=f["bg"]).pack(side=tk.RIGHT, padx=10)

        self.draw_projection(all_strokes)

    def draw_projection(self, all_strokes):
        self.canvas.delete("projection")
        result = self.recognizer.autocomplete(all_strokes)
        if result:
            char, future_points = result
            if future_points:
                for i in range(len(future_points) - 1):
                    x1, y1 = future_points[i]
                    x2, y2 = future_points[i+1]
                    self.canvas.create_line(x1, y1, x2, y2, fill="#3498db", width=3, dash=(4, 4), tags="projection")