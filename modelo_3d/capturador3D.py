import json
import math
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox

# Constantes visuales
BG, CARD, TEXT, MUTED = "#f0f4f8", "#ffffff", "#1e293b", "#64748b"
ACCENT, SURFACE, SURFACE_EDGE, INK = "#2563eb", "#f8fafc", "#e2e8f0", "#0f172a"

class Capturador3D(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("Capturador de Dataset 3D - Alfabeto Completo")
        self.geometry("900x750")
        self.configure(fg_color=BG)

        self.size = 550
        self.strokes = []      
        self.items = []        
        self.drawing = False
        self.disp = []         

        # Generar el alfabeto de la A a la Z + Ñ (en minúsculas)
        letras_str = "abcdefghijklmnñopqrstuvwxyz"
        self.dataset = {letra: [] for letra in letras_str}

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

        ctk.CTkButton(left, text="Borrar Trazo Actual", fg_color="#ef4444", hover_color="#dc2626",
                      font=ctk.CTkFont(weight="bold"), command=self.clear_current).pack(pady=(0, 20))

        # Contenedor derecho: Botones de guardado con SCROLL
        right = ctk.CTkFrame(self, fg_color=CARD, corner_radius=20, width=280)
        right.pack(side="right", fill="y", padx=(0, 20), pady=20)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="1. Dibuja\n2. Guarda", 
                     font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT).pack(pady=(20, 10))

        # Frame con scroll para los 27 botones
        scroll_frame = ctk.CTkScrollableFrame(right, width=240, height=500, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.labels_count = {}
        letras = list(self.dataset.keys())
        
        # Crear botones en cuadrícula (3 columnas)
        for i, letra in enumerate(letras):
            row = i // 3
            col = i % 3
            
            # AQUÍ ESTABA EL ERROR DE INDENTACIÓN (Ya está corregido)
            btn = ctk.CTkButton(scroll_frame, text=letra, width=65, height=35,
                                fg_color=ACCENT, command=lambda l=letra: self.save_sample(l))
            btn.grid(row=row, column=col, padx=5, pady=5)
            
            lbl = ctk.CTkLabel(scroll_frame, text="0", text_color=MUTED, font=ctk.CTkFont(size=12, weight="bold"))
            lbl.grid(row=row, column=col, sticky="se", padx=(0, 10), pady=(0, 2))
            self.labels_count[letra] = lbl

        ctk.CTkButton(right, text="💾 Exportar JSON", fg_color="#10b981", hover_color="#059669",
                      height=50, font=ctk.CTkFont(size=16, weight="bold"),
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
        cx = cy = self.size / 2
        r = self.size / 2 - 10
        
        nx = (mx - cx) / r
        ny = -(my - cy) / r 
        
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
            
        self.dataset[letra].append([list(p) for p in self.strokes])
        
        count = len(self.dataset[letra])
        self.labels_count[letra].configure(text=str(count), text_color="#10b981")
        
        self.clear_current()

    def export_json(self):
        vacias = [k for k, v in self.dataset.items() if len(v) == 0]
        if vacias:
            messagebox.showerror("Faltan datos", f"Aún no has dibujado nada para:\n{', '.join([v.upper() for v in vacias])}")
            return

        out_path = "glifo-3d.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2)
            
        messagebox.showinfo("Éxito", f"¡Felicidades!\nDataset 3D con 27 letras guardado en {out_path}")

if __name__ == "__main__":
    app = Capturador3D()
    app.mainloop() # <-- ESTO FALTABA