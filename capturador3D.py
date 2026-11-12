import json
import math
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox

# Constantes visuales (similares a tu GUI)
BG, CARD, TEXT, MUTED = "#f0f4f8", "#ffffff", "#1e293b", "#64748b"
ACCENT, SURFACE, SURFACE_EDGE, INK = "#2563eb", "#f8fafc", "#e2e8f0", "#0f172a"

class Capturador3D(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("Capturador de Dataset 3D")
        self.geometry("800x700")
        self.configure(fg_color=BG)

        self.size = 500
        self.strokes = []      # Trazos actuales en 3D
        self.items = []        # Objetos del canvas
        self.drawing = False
        self.disp = []         # Puntos en 2D para dibujar

        # El dataset final
        self.dataset = {"a": [], "b": [], "c": [], "x": [], "y": []}

        self._build_ui()

    def _build_ui(self):
        # Contenedor izquierdo: Lienzo
        left = ctk.CTkFrame(self, fg_color=CARD, corner_radius=20)
        left.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.canvas = tk.Canvas(left, width=self.size, height=self.size, bg=CARD,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=20, pady=20)
        self._draw_sphere()
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        ctk.CTkButton(left, text="Borrar Trazo", fg_color="#ef4444", hover_color="#dc2626",
                      command=self.clear_current).pack(pady=(0, 20))

        # Contenedor derecho: Botones de guardado
        right = ctk.CTkFrame(self, fg_color=CARD, corner_radius=20, width=250)
        right.pack(side="right", fill="y", padx=(0, 20), pady=20)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="1. Dibuja una letra\n2. Guárdala", 
                     font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT).pack(pady=(20, 10))

        # Botones para cada letra
        letras = ["a", "b", "c", "x", "y"]
        self.labels_count = {}

        for letra in letras:
            frame = ctk.CTkFrame(right, fg_color="transparent")
            frame.pack(fill="x", padx=20, pady=5)
            
            btn = ctk.CTkButton(frame, text=f"Guardar '{letra}'", width=120,
                                fg_color=ACCENT, command=lambda l=letra: self.save_sample(l))
            btn.pack(side="left")
            
            lbl = ctk.CTkLabel(frame, text="0", text_color=MUTED, font=ctk.CTkFont(weight="bold"))
            lbl.pack(side="right")
            self.labels_count[letra] = lbl

        ctk.CTkButton(right, text="Exportar JSON", fg_color="#10b981", hover_color="#059669",
                      height=45, font=ctk.CTkFont(size=15, weight="bold"),
                      command=self.export_json).pack(side="bottom", padx=20, pady=20, fill="x")

    def _draw_sphere(self):
        c, s = self.canvas, self.size
        cx = cy = s / 2
        r = s / 2 - 10
        
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=SURFACE, outline=SURFACE_EDGE, width=2)
        
        for offset in [0.35, 0.7]:
            c.create_oval(cx - r * offset, cy - r, cx + r * offset, cy + r, outline=SURFACE_EDGE, width=1)
            c.create_oval(cx - r, cy - r * offset, cx + r, cy + r * offset, outline=SURFACE_EDGE, width=1)
            
        c.create_line(cx, cy - r, cx, cy + r, fill=SURFACE_EDGE, width=1)
        c.create_line(cx - r, cy, cx + r, cy, fill=SURFACE_EDGE, width=1)

    def _map_to_sphere(self, mx, my):
        """Convierte píxeles (mx, my) a coordenadas 3D en la esfera."""
        cx = cy = self.size / 2
        r = self.size / 2 - 10
        
        nx = (mx - cx) / r
        ny = -(my - cy) / r # Invertir Y
        
        dist_sq = nx**2 + ny**2
        if dist_sq > 1.0:
            dist = np.sqrt(dist_sq)
            nx /= dist
            ny /= dist
            nz = 0.0
        else:
            nz = np.sqrt(1.0 - dist_sq)
            
        return (nx, ny, nz)

    def on_press(self, e):
        self.drawing = True
        p3d = self._map_to_sphere(e.x, e.y)
        self.strokes.append([p3d])
        self.disp = [(e.x, e.y)]
        
        item = self.canvas.create_oval(e.x-3, e.y-3, e.x+3, e.y+3, fill=INK, outline=INK, tags="ink")
        self.items.append(item)

    def on_move(self, e):
        if not self.drawing or not self.strokes:
            return
            
        p3d = self._map_to_sphere(e.x, e.y)
        stroke = self.strokes[-1]
        
        # Distancia mínima para guardar el punto (evitar ruido)
        if np.linalg.norm(np.array(p3d) - np.array(stroke[-1])) * (self.size/2) < 5:
            return
            
        stroke.append(p3d)
        self.disp.append((e.x, e.y))
        
        flat = [c for p in self.disp for c in p]
        if len(self.disp) == 2:
            self.canvas.delete(self.items[-1])
            self.items[-1] = self.canvas.create_line(*flat, fill=INK, width=5, smooth=True,
                                                     splinesteps=8, capstyle="round", tags="ink")
        else:
            self.canvas.coords(self.items[-1], *flat)

    def on_release(self, e):
        self.drawing = False

    def clear_current(self):
        self.strokes = []
        self.items = []
        self.disp = []
        self.canvas.delete("ink")

    def save_sample(self, letra):
        if not self.strokes:
            messagebox.showwarning("Vacío", "Dibuja algo primero antes de guardar.")
            return
            
        # Guardamos una copia profunda de los trazos
        self.dataset[letra].append([list(p) for p in self.strokes])
        
        # Actualizar contador visual
        count = len(self.dataset[letra])
        self.labels_count[letra].configure(text=str(count))
        
        # Limpiar para el siguiente
        self.clear_current()

    def export_json(self):
        # Validar que haya al menos 1 muestra por letra
        vacias = [k for k, v in self.dataset.items() if len(v) == 0]
        if vacias:
            messagebox.showerror("Faltan datos", f"Faltan muestras para: {', '.join(vacias)}")
            return

        out_path = "glifo-3d.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2)
            
        messagebox.showinfo("Éxito", f"Dataset guardado en {out_path}\nRevisa la carpeta de tu proyecto.")

if __name__ == "__main__":
    app = Capturador3D()
    app.mainloop()