"""
GUI.py
Interfaz gráfica del modo 2D con diseño moderno (CustomTkinter).
"""
import customtkinter as ctk
from tkinter import messagebox
from Recognizer import Recognizer
from FeatureExtractor import Extractor2D
from ModelRepository import ModelRepository
import os
import tkinter as tk

# Configuración de apariencia global
ctk.set_appearance_mode("Dark")  # Modos: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Temas: "blue", "green", "dark-blue"

class App2D(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Reconocedor HMM 2D - Procesos Estocásticos")
        self.geometry("1100x700")
        
        # --- Lógica (Backend) ---
        # IMPORTANTE: min_dist debe coincidir con el de train_2d.py
        self.extractor = Extractor2D(k_dir=16, min_dist=8.0) 
        model_path = os.path.join("modelo_2d", "models_2d.json")
        
        try:
            repo = ModelRepository(model_path)
            classifier = repo.load()
            self.recognizer = Recognizer(self.extractor, classifier)
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el modelo en {model_path}.\nEjecuta train_2d.py primero.")
            self.destroy()
            return

        self.strokes = []
        self.current_stroke = []
        self.is_drawing = False

        self._build_ui()

    def _build_ui(self):
        # Contenedor principal
        self.grid_columnconfigure(0, weight=3) # Lienzo más grande
        self.grid_columnconfigure(1, weight=1) # Panel más pequeño
        self.grid_rowconfigure(0, weight=1)

        # --- SECCIÓN IZQUIERDA: LIENZO ---
        left_frame = ctk.CTkFrame(self, corner_radius=15)
        left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Título de sección
        ctk.CTkLabel(left_frame, text="Lienzo de Dibujo", font=("Arial", 18, "bold")).pack(pady=(15, 5))

        # Canvas (Tkinter nativo dentro de CTk)
        self.canvas = tk.Canvas(left_frame, bg="#2b2b2b", highlightthickness=0, cursor="pencil")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Botones de control
        btn_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        btn_frame.pack(side=tk.BOTTOM, pady=15)
        
        ctk.CTkButton(btn_frame, text="Limpiar Lienzo", command=self.clear_canvas, 
                      fg_color="#e74c3c", hover_color="#c0392b", font=("Arial", 14, "bold")).pack(side=tk.LEFT, padx=5)

        # --- SECCIÓN DERECHA: PANEL DE PROBABILIDADES ---
        right_frame = ctk.CTkFrame(self, corner_radius=15)
        right_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        
        ctk.CTkLabel(right_frame, text="Clasificación en Tiempo Real", font=("Arial", 16, "bold")).pack(pady=(20, 10))
        
        self.prob_frame = ctk.CTkScrollableFrame(right_frame, fg_color="transparent")
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def start_draw(self, event):
        self.is_drawing = True
        self.current_stroke = [(event.x, event.y)]
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="white", outline="")

    def draw(self, event):
        if self.is_drawing:
            x, y = event.x, event.y
            if self.current_stroke:
                px, py = self.current_stroke[-1]
                self.canvas.create_line(px, py, x, y, fill="white", width=3, smooth=True)
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
        # Limpiar panel derecho
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

        # Mostrar Top 5 letras
        for i, (char, prob) in enumerate(posteriors[:5]):
            if prob < 0.01: break
            
            # Frame por cada letra
            f = ctk.CTkFrame(self.prob_frame, corner_radius=8, fg_color=("#3a3a3a", "#2b2b2b"))
            f.pack(fill="x", pady=5)
            
            # Si es la más probable, resaltarla
            if i == 0:
                f.configure(fg_color=("#1f538d", "#14375e")) # Azul oscuro
                txt_color = "white"
            else:
                txt_color = ("gray80", "gray80")
            
            # AQUÍ ESTÁ EL CAMBIO: Quitamos el .upper() para que salga en minúscula
            ctk.CTkLabel(f, text=char, font=("Arial", 22, "bold"), text_color=txt_color, width=40).pack(side="left", padx=15, pady=10)
            ctk.CTkLabel(f, text=f"{prob*100:.1f}%", font=("Arial", 16), text_color=txt_color).pack(side="right", padx=15)

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
                    # Línea de autocompletado color verde neón
                    self.canvas.create_line(x1, y1, x2, y2, fill="#00ffcc", width=4, dash=(6, 4), tags="projection")

if __name__ == "__main__":
    app = App2D()
    app.mainloop()