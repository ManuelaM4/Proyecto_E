import os
import tkinter as tk
import customtkinter as ctk
import numpy as np
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFont
import random
import math

from Recognizer import Recognizer
from FeatureExtractor import Extractor2D
from ModelRepository import ModelRepository

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_2D = os.path.join(BASE, "modelo_2d", "models_2d.json")
MIN_DIST_2D = 15.0      # DEBE ser igual al min_dist de modelo_2d/train_2d.py

# ----------------------------------------------------------------- estilo y parámetros
# ----------------------------------------------------------------- estilo y parámetros
BG, CARD, TEXT, MUTED = "#f0f4f8", "#ffffff", "#1e293b", "#64748b"
ACCENT, ACCENT_SOFT, TILE, BAR_OFF = "#2563eb", "#60a5fa", "#e0f2fe", "#e2e8f0"
INK, SURFACE, SURFACE_EDGE, DANGER = "#0f172a", "#f8fafc", "#e2e8f0", "#ef4444"
SUCCESS, ROW_FIRST = "#10b981", "#d1fae5"

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "iconos")
DATA_SIZE = 1000.0      # tamaño del lienzo con el que se capturó el dataset
BRUSH = 5               # grosor del pincel (px)
MIN_STEP = 2.0          # distancia mínima entre puntos capturados (px)
FOLLOW = 0.55           # suavizado del trazo: 1 = sin suavizar, menor = más suave
UPDATE_MS = 50          # refresco máximo del panel (~20 fps)
TEMPERATURE = 4.0       # solo suaviza la visualización de las probabilidades
TOP_N = 8               # letras mostradas en el listado


# ======================================================================
# Ventana A · Menú principal
# ======================================================================


class MainMenu(ctk.CTkFrame):
    def __init__(self, master, on_2d, on_3d):
        super().__init__(master, fg_color=BG)
        
        # --- 1. CANVAS DE FONDO ---
        self.bg_canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # Redibujar el fondo cada vez que la ventana cambie de tamaño
        self.bg_canvas.bind("<Configure>", self._on_canvas_resize)
        
        # --- 2. CONTENEDOR PRINCIPAL ---
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.place(relx=0.5, rely=0.45, anchor="center")

        # --- 3. CABECERA ---
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

        # --- 4. TARJETAS ---
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
            btn_text="En Construcción", cmd=on_3d, 
            theme_color=SUCCESS, bg_color="#f2fdf6", is_active=False
        )
        self._card_3d.pack(side="left", padx=20)

        # Footer
        ctk.CTkLabel(self, text="Procesos Estocásticos · 603504 · 5.º Semestre", text_color=MUTED,
                     font=ctk.CTkFont(family="Segoe UI", size=12)).pack(side="bottom", pady=20)

    def _on_canvas_resize(self, event):
        """Este evento se dispara cuando el canvas ya tiene su tamaño real."""
        self._draw_background_pattern(event.width, event.height)

    def _draw_background_pattern(self, w, h):
        """Dibuja un patrón estructurado de letras y ondas suaves en el fondo."""
        self.bg_canvas.delete("all") 
        
        # 1. Cuadrícula elegante de símbolos matemáticos
        letras = ["a", "b", "c", "x", "y", "z", "π", "λ", "α", "β", "∑", "∫"]
        
        # AQUI ESTA EL CAMBIO: Usamos tonos un poco más oscuros para que resalten
        colores_fondo = ["#cbd5e1", "#bfdbfe", "#bbf7d0", "#fecaca"] 
        
        spacing = 160
        idx = 0
        for x in range(40, w + 100, spacing):
            for y in range(40, h + 100, spacing):
                char = letras[idx % len(letras)]
                color = colores_fondo[idx % len(colores_fondo)]
                size = 50 + (idx % 4) * 15  # Tamaños ligeramente variables
                
                # Truco de diseño: desfase en filas pares para que parezca un panal
                offset_x = (spacing // 2) if (y // spacing) % 2 == 0 else 0
                
                self.bg_canvas.create_text(x + offset_x, y, text=char, fill=color,
                                           font=("Segoe UI", size, "bold"))
                idx += 1

        # 2. Ondas matemáticas suaves (ahora más visibles)
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
        # MARCO EXTERIOR (Para simular sombra)
        shadow = ctk.CTkFrame(parent, fg_color="#e2e8f0", corner_radius=28, width=324, height=454)
        shadow.pack_propagate(False)
        
        # Marco de la tarjeta
        card = ctk.CTkFrame(shadow, fg_color=bg_color, corner_radius=26, width=320, height=450,
                            border_width=2, border_color=bg_color)
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)

        def on_enter(e):
            card.configure(border_color=theme_color)
            btn.configure(fg_color=theme_color if is_active else DANGER)
            btn.configure(text_color="white")
            canvas.itemconfigure("trazo", width=6)

        def on_leave(e):
            card.configure(border_color=bg_color)
            if is_active:
                btn.configure(fg_color=theme_color, text_color="white")
            else:
                btn.configure(fg_color="#fee2e2", text_color=DANGER)
            canvas.itemconfigure("trazo", width=4)

        for widget in (card, shadow):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        # --- DIBUJO ILUSTRATIVO CON CANVAS ---
        canvas = tk.Canvas(card, width=240, height=200, bg=bg_color, highlightthickness=0)
        canvas.pack(pady=(30, 10))
        
        if title == "Modo 2D":
            canvas.create_line(30, 180, 210, 180, fill="#bfdbfe", width=3) 
            canvas.create_line(30, 20, 30, 180, fill="#bfdbfe", width=3)  
            
            pts = [(50, 150), (70, 110), (100, 80), (130, 100), (160, 60), (190, 70)]
            for i in range(len(pts)-1):
                canvas.create_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], 
                                   fill="#bfdbfe", width=2, dash=(2, 4))
            
            canvas.create_line(*[c for p in pts for c in p], fill=theme_color, width=4, 
                               smooth=True, capstyle="round", tags="trazo")
            
            canvas.create_oval(185, 65, 195, 75, fill=DANGER, outline="white", width=2)

        else:
            canvas.create_oval(60, 40, 180, 160, outline="#bbf7d0", width=3, fill="#dcfce7")
            canvas.create_arc(60, 60, 180, 140, start=0, extent=180, style="arc", outline="#86efac", width=4)
            canvas.create_arc(60, 60, 180, 140, start=180, extent=180, style="arc", outline="#86efac", width=4, dash=(4, 4))
            canvas.create_line(90, 50, 120, 110, 150, 70, fill=theme_color, width=4, smooth=True, capstyle="round", tags="trazo")
            canvas.create_oval(145, 65, 155, 75, fill=DANGER, outline="white", width=2)

        # --- TEXTOS ---
        ctk.CTkLabel(card, text=title, text_color=TEXT,
                     font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold")).pack(pady=(10, 2))
        
        ctk.CTkLabel(card, text=subtitle, text_color=MUTED,
                     font=ctk.CTkFont(family="Segoe UI", size=14)).pack()

        # --- BOTÓN ---
        btn_color = theme_color if is_active else "#fee2e2"
        txt_color = "white" if is_active else DANGER
        hover_color = "#1d4ed8" if is_active else DANGER

        btn = ctk.CTkButton(card, text=btn_text, width=220, height=48, corner_radius=24,
                            fg_color=btn_color, hover_color=hover_color,
                            text_color=txt_color, 
                            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
                            command=cmd)
        btn.pack(side="bottom", pady=30)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        return shadow 

# ======================================================================
# Ventana C · Modo 3D (pendiente)
# ======================================================================
class Window3D(ctk.CTkFrame):
    def __init__(self, master, on_back):
        super().__init__(master, fg_color=BG)
        ctk.CTkButton(self, text="←  Menú", width=90, height=34, corner_radius=17,
                      fg_color=CARD, text_color=TEXT, hover_color="#e6e8f2",
                      command=on_back).place(x=24, y=18)
        card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22, width=460, height=240)
        card.place(relx=0.5, rely=0.5, anchor="center")
        card.pack_propagate(False)
        ctk.CTkLabel(card, text="Modo 3D", text_color=TEXT,
                     font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(50, 6))
        ctk.CTkLabel(card, text="En construcción: aquí irá la esfera para dibujar\n"
                                "y el panel de probabilidades.",
                     text_color=MUTED, font=ctk.CTkFont(size=14), justify="center").pack()


# ======================================================================
# Ventana B · Modo 2D
# ======================================================================
class Window2D(ctk.CTkFrame):
    def __init__(self, master, recognizer, on_back):
        super().__init__(master, fg_color=BG)
        self.rec, self.on_back = recognizer, on_back
        self.size = int(max(380, min(560, self.winfo_screenheight() - 300)))
        self.S = DATA_SIZE / self.size
        self.strokes = []          # trazos crudos en px: [[(x, y), ...], ...]
        self.items = []            # un objeto de canvas por trazo (para deshacer)
        self.disp = []             # puntos suavizados del trazo actual
        self.cur_kind = None       # "dot" | "line"
        self.drawing = False
        self._pending = None
        self._last_key = None
        self.proj_item = None
        self.icons, self.slots = {}, []
        self._build()
        self.reset_panel()
        top = self.winfo_toplevel()
        top.bind("<Control-z>", lambda e: self.undo() if self.winfo_ismapped() else None)
        top.bind("<Delete>", lambda e: self.clear_canvas() if self.winfo_ismapped() else None)

    # ================================================================ interfaz
    @staticmethod
    def _make_icon_image(letter, px=128):
        S = 4
        n = px * S
        img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, n - 1, n - 1), radius=int(n * 0.24), fill=TILE)
        font = None
        for name in ("segoeuib.ttf", "arialbd.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf",
                     "Arial Bold.ttf", "Helvetica.ttc", "LiberationSans-Bold.ttf"):
            try:
                font = ImageFont.truetype(name, int(n * 0.62))
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=int(n * 0.62))
            except TypeError:
                font = ImageFont.load_default()
        bb = d.textbbox((0, 0), letter, font=font)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((n - w) / 2 - bb[0], (n - h) / 2 - bb[1]), letter, font=font, fill=ACCENT)
        return img.resize((px, px), Image.LANCZOS)

    def _icon(self, letter):
        if letter not in self.icons:
            path = os.path.join(ICON_DIR, f"{letter}.png")
            if letter == "_vacio":
                img = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
            elif os.path.exists(path):
                img = Image.open(path)
            else:
                img = self._make_icon_image(letter)
            self.icons[letter] = ctk.CTkImage(light_image=img, dark_image=img, size=(40, 40))
        return self.icons[letter]

    def _build(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---------- cabecera ----------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(18, 10))
        ctk.CTkButton(header, text="←  Menú", width=90, height=34, corner_radius=17,
                      fg_color=CARD, text_color=TEXT, hover_color="#e6e8f2",
                      command=self.on_back).pack(side="left")
        ctk.CTkLabel(header, text="Reconocedor HMM", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT).pack(side="left", padx=(16, 8))
        ctk.CTkLabel(header, text="Modo 2D", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="white", fg_color=ACCENT, corner_radius=10,
                     width=70, height=26).pack(side="left")

        # ---------- tarjeta izquierda: herramientas + lienzo ----------
        left = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22)
        left.grid(row=1, column=0, sticky="n", padx=(24, 12), pady=(0, 24))

        tools = ctk.CTkFrame(left, fg_color="transparent")
        tools.pack(fill="x", padx=18, pady=(16, 10))
        ctk.CTkButton(tools, text="↶  Deshacer", width=110, height=36, corner_radius=18,
                      fg_color=TILE, text_color=TEXT, hover_color="#dcdff5",
                      command=self.undo).pack(side="left", padx=(0, 8))
        ctk.CTkButton(tools, text="Limpiar lienzo", width=130, height=36, corner_radius=18,
                      fg_color=DANGER, hover_color="#c93a3f", text_color="white",
                      command=self.clear_canvas).pack(side="left", padx=(0, 8))
        ctk.CTkButton(tools, text="↻  Reiniciar captura", width=150, height=36, corner_radius=18,
                      fg_color=ACCENT, hover_color="#4a4ac0", text_color="white",
                      command=self.restart_capture).pack(side="left")

        self.canvas = tk.Canvas(left, width=self.size, height=self.size, bg=CARD,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=18, pady=(0, 8))
        self._draw_surface()
        self.ring = self.canvas.create_oval(0, 0, 0, 0, outline=ACCENT, width=2,
                                            state="hidden", tags="ring")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", lambda e: self._move_ring(e.x, e.y))
        self.canvas.bind("<Leave>", lambda e: self.canvas.itemconfigure(self.ring, state="hidden"))

        self.lbl_status = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=13), text_color=MUTED)
        self.lbl_status.pack(pady=(0, 14))

        # ---------- tarjeta derecha: resultado + Top N ----------
        right = ctk.CTkFrame(self, fg_color=CARD, corner_radius=22)
        right.grid(row=1, column=1, sticky="nsew", padx=(12, 24), pady=(0, 24))
        right.grid_columnconfigure(0, weight=1)

        hero = ctk.CTkFrame(right, fg_color=ACCENT, corner_radius=18)
        hero.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        self.lbl_hero = ctk.CTkLabel(hero, text="?", width=88, height=88, corner_radius=16,
                                     fg_color="white", text_color=ACCENT,
                                     font=ctk.CTkFont(size=50, weight="bold"))
        self.lbl_hero.pack(side="left", padx=16, pady=16)
        info = ctk.CTkFrame(hero, fg_color="transparent")
        info.pack(side="left", padx=6)
        ctk.CTkLabel(info, text="Letra más probable", text_color="#dcdcff",
                     font=ctk.CTkFont(size=12)).pack(anchor="w")
        self.lbl_hero_p = ctk.CTkLabel(info, text="—", text_color="white",
                                       font=ctk.CTkFont(size=32, weight="bold"))
        self.lbl_hero_p.pack(anchor="w")

        ctk.CTkLabel(right, text=f"Top {TOP_N} · P(carácter | O₁:t)", text_color=MUTED,
                     font=ctk.CTkFont(size=12, weight="bold")
                     ).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 4))

        lst = ctk.CTkFrame(right, fg_color="transparent")
        lst.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 16))
        lst.grid_columnconfigure(0, weight=1)
        for i in range(TOP_N):
            row = ctk.CTkFrame(lst, fg_color=ROW_FIRST if i == 0 else "transparent", corner_radius=14)
            row.grid(row=i, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(3, weight=1)
            rank = ctk.CTkLabel(row, text=str(i + 1), width=22, text_color=ACCENT if i == 0 else MUTED,
                                font=ctk.CTkFont(size=13, weight="bold"))
            rank.grid(row=0, column=0, padx=(10, 2), pady=5)
            icon = ctk.CTkLabel(row, text="", image=self._icon("_vacio"), width=40, height=40)
            icon.grid(row=0, column=1, padx=(2, 8))
            name = ctk.CTkLabel(row, text="·", width=26, text_color=ACCENT if i == 0 else TEXT,
                                font=ctk.CTkFont(size=22, weight="bold"))
            name.grid(row=0, column=2, padx=(0, 10))
            bar = ctk.CTkProgressBar(row, height=10, corner_radius=5, fg_color="#e9ebf6",
                                     progress_color=ACCENT if i == 0 else (ACCENT_SOFT if i < 3 else BAR_OFF))
            bar.grid(row=0, column=3, sticky="ew")
            bar.set(0)
            pct = ctk.CTkLabel(row, text="—", width=64, anchor="e", text_color=TEXT,
                               font=ctk.CTkFont(size=14, weight="bold" if i == 0 else "normal"))
            pct.grid(row=0, column=4, padx=(10, 12))
            self.slots.append({"letter": None, "icon": icon, "name": name, "bar": bar, "pct": pct})

    # ================================================================ lienzo
    def _draw_surface(self):
        c, s, r = self.canvas, self.size, 22
        x1 = y1 = 2
        x2 = y2 = s - 2
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
               x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        c.create_polygon(pts, smooth=True, fill=SURFACE, outline=SURFACE_EDGE, width=2)
        self._show_hint()

    def _show_hint(self):
        self.canvas.delete("hint")
        self.canvas.create_text(self.size / 2, self.size / 2, text="Dibuja aquí una letra minúscula",
                                fill="#b3b8d1", font=("Helvetica", 15), tags="hint")

    def _move_ring(self, x, y):
        r = BRUSH + 1
        self.canvas.coords(self.ring, x - r, y - r, x + r, y + r)
        self.canvas.itemconfigure(self.ring, state="normal")
        self.canvas.tag_raise("ring")

    # ================================================================ ratón
    def on_press(self, e):
        self.drawing = True
        self.canvas.delete("hint")
        self.strokes.append([(e.x, e.y)])
        self.disp = [(e.x, e.y)]
        r = BRUSH / 2.0
        item = self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r,
                                       fill=INK, outline=INK, tags="ink")
        self.items.append(item)
        self.cur_kind = "dot"
        self._move_ring(e.x, e.y)
        self.schedule_update()

    def on_move(self, e):
        self._move_ring(e.x, e.y)
        if not self.drawing or not self.strokes:
            return
        stroke = self.strokes[-1]
        lx, ly = stroke[-1]
        if (e.x - lx) ** 2 + (e.y - ly) ** 2 < MIN_STEP ** 2:
            return
        stroke.append((e.x, e.y))                       # punto crudo -> reconocimiento
        px, py = self.disp[-1]                          # punto suavizado -> pantalla
        self.disp.append((px + (e.x - px) * FOLLOW, py + (e.y - py) * FOLLOW))
        self._update_line()
        self.schedule_update()

    def on_release(self, e):
        if self.drawing and self.strokes and self.cur_kind == "line":
            self.disp.append(self.strokes[-1][-1])      # cierra el trazo en el último punto real
            self._update_line()
        self.drawing = False
        self.do_update()

    def _update_line(self):
        """Un solo objeto de línea por trazo: se alarga, nunca se redibuja por pedazos."""
        flat = [c for p in self.disp for c in p]
        if self.cur_kind == "dot":
            self.canvas.delete(self.items[-1])
            self.items[-1] = self.canvas.create_line(*flat, fill=INK, width=BRUSH, smooth=True,
                                                     splinesteps=8, capstyle="round",
                                                     joinstyle="round", tags="ink")
            self.cur_kind = "line"
        else:
            self.canvas.coords(self.items[-1], *flat)
        self.canvas.tag_raise("ring")

    # ================================================================ reconocimiento
    def _data_strokes(self):
        return [[(x * self.S, y * self.S) for x, y in s] for s in self.strokes]

    def _ranking(self, data):
        """P(c | O_1:t) por Bayes con prior uniforme, usando log-verosimilitudes Forward.
        La temperatura solo suaviza la visualización (T=1 es el posterior exacto)."""
        seq = self.rec.extractor.extract(data)
        if not seq:
            return []
        models = self.rec.classifier.models
        chars = list(models.keys())
        ll = np.array([models[c].log_likelihood(seq) for c in chars]) / TEMPERATURE
        ll -= ll.max()                                  # log-sum-exp estable
        p = np.exp(ll)
        p /= p.sum()
        order = np.argsort(-p)
        return [(chars[i], float(p[i])) for i in order]

    def schedule_update(self):
        if self._pending is None:
            self._pending = self.after(UPDATE_MS, self.do_update)

    def do_update(self):
        self._pending = None
        if not self.strokes:
            return
        data = self._data_strokes()
        ranking = self._ranking(data)
        if not ranking:
            self._hide_projection()
            return
        top = [(l, round(p, 3)) for l, p in ranking[:TOP_N]]
        if top != self._last_key:                       # solo se repinta si algo cambió
            self._last_key = top
            self._show_top(ranking[:TOP_N])
        self._draw_projection(data)

    def _show_top(self, top):
        for slot, (letter, p) in zip(self.slots, top):
            if slot["letter"] != letter:                # el icono solo cambia si cambia la letra
                slot["letter"] = letter
                slot["icon"].configure(image=self._icon(letter))
                slot["name"].configure(text=letter)
            slot["bar"].set(float(p))
            slot["pct"].configure(text=f"{100 * p:.1f}%")
        best, pb = top[0]
        self.lbl_hero.configure(text=best)
        self.lbl_hero_p.configure(text=f"{100 * pb:.1f}%")
        self.lbl_status.configure(
            text=f"Autocompletando como «{best}»  ·  línea punteada = trayectoria estimada")

    def _hide_projection(self):
        if self.proj_item is not None:
            self.canvas.itemconfigure(self.proj_item, state="hidden")

    def _draw_projection(self, data):
        res = self.rec.autocomplete(data)
        if not res or not res[1]:
            self._hide_projection()
            return
        x, y = self.strokes[-1][-1]
        path = [x, y]
        for px, py in res[1]:                           # unidades del dataset -> píxeles
            path += [px / self.S, py / self.S]
        if len(path) < 4:
            self._hide_projection()
            return
        if self.proj_item is None:
            self.proj_item = self.canvas.create_line(*path, fill=ACCENT_SOFT, width=4, dash=(2, 7),
                                                     capstyle="round", smooth=True, tags="proj")
        else:
            self.canvas.coords(self.proj_item, *path)
            self.canvas.itemconfigure(self.proj_item, state="normal")
        self.canvas.tag_raise("ink")
        self.canvas.tag_raise("ring")

    # ================================================================ controles
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
        """Quita el último trazo."""
        if not self.strokes:
            return
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
        """Limpiar lienzo: borra todo lo dibujado y la proyección."""
        self.strokes, self.items, self.disp = [], [], []
        self.drawing = False
        self.canvas.delete("ink")
        self._hide_projection()
        self.reset_panel()
        self._show_hint()

    def restart_capture(self):
        """Reiniciar captura: vuelve al estado inicial (lienzo + probabilidades)."""
        self.clear_canvas()


# ======================================================================
# Aplicación principal
# ======================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("Reconocedor HMM · Procesos Estocásticos")
        w, h = 1120, 740
        x = max(0, (self.winfo_screenwidth() - w) // 2)
        y = max(0, (self.winfo_screenheight() - h) // 2 - 20)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1000, 660)
        self.configure(fg_color=BG)

        self.recognizer_2d = self._load_model_2d()      # la guía pide cargar los modelos al iniciar
        self.current = None
        self.window2d = None
        self.window3d = None
        self.menu = MainMenu(self, on_2d=self.open_2d, on_3d=self.open_3d)
        self.show(self.menu)

    @staticmethod
    def _load_model_2d():
        try:
            classifier = ModelRepository(MODEL_2D).load()
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el modelo:\n{MODEL_2D}\n"
                                          "Entrénalo con:  python modelo_2d/train_2d.py")
            return None
        return Recognizer(Extractor2D(k_dir=16, min_dist=MIN_DIST_2D), classifier)

    def show(self, frame):
        if self.current is not None:
            self.current.pack_forget()
        self.current = frame
        frame.pack(fill="both", expand=True)

    def go_menu(self):
        self.show(self.menu)

    def open_2d(self):
        if self.recognizer_2d is None:
            self.recognizer_2d = self._load_model_2d()
            if self.recognizer_2d is None:
                return
        if self.window2d is None:
            self.window2d = Window2D(self, self.recognizer_2d, on_back=self.go_menu)
        self.show(self.window2d)

    def open_3d(self):
        if self.window3d is None:
            self.window3d = Window3D(self, on_back=self.go_menu)
        self.show(self.window3d)

if __name__ == "__main__":
    app = App()
    app.mainloop()