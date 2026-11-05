import customtkinter as ctk
import tkinter as tk
import os
from Recognizer import Recognizer
from FeatureExtractor import Extractor2D
from ModelRepository import ModelRepository
from tkinter import messagebox

# Configuración de apariencia global (Modo claro como en tu imagen)
ctk.set_appearance_mode("Light")  
ctk.set_default_color_theme("blue")  

class VentanaMenu(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=1, column=0, pady=20)
        
        ctk.CTkLabel(title_frame, text="UNIVERSIDAD DE LOS LLANOS", font=("Arial", 14, "bold"), text_color="gray").pack()
        ctk.CTkLabel(title_frame, text="Reconocedor de caracteres", font=("Arial", 32, "bold")).pack(pady=(10, 0))
        ctk.CTkLabel(title_frame, text="con Cadenas Ocultas de Markov", font=("Arial", 18), text_color="#6C5CE7").pack()
        ctk.CTkLabel(title_frame, text="Procesos Estocásticos - 603504", font=("Arial", 12), text_color="gray").pack(pady=(10, 0))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, pady=30)
        
        card_2d = ctk.CTkFrame(btn_frame, width=250, height=200, corner_radius=15)
        card_2d.pack(side="left", padx=20)
        card_2d.pack_propagate(False)
        ctk.CTkLabel(card_2d, text="Modo 2D", font=("Arial", 22, "bold")).pack(pady=(30, 5))
        ctk.CTkLabel(card_2d, text="Trazos sobre un plano\ncartesiano", font=("Arial", 12), text_color="gray").pack()
        ctk.CTkButton(card_2d, text="Empezar", font=("Arial", 14, "bold"), command=self.app.mostrar_canvas).pack(side="bottom", pady=20, padx=20, fill="x")

        card_3d = ctk.CTkFrame(btn_frame, width=250, height=200, corner_radius=15)
        card_3d.pack(side="left", padx=20)
        card_3d.pack_propagate(False)
        ctk.CTkLabel(card_3d, text="Modo 3D", font=("Arial", 22, "bold"), text_color="gray").pack(pady=(30, 5))
        ctk.CTkLabel(card_3d, text="Trazos sobre una esfera", font=("Arial", 12), text_color="gray").pack()
        ctk.CTkButton(card_3d, text="Próximamente", font=("Arial", 14, "bold"), state="disabled", fg_color="gray").pack(side="bottom", pady=20, padx=20, fill="x")


class VentanaCanvas2D(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        
        self.extractor = Extractor2D(k_dir=16, min_dist=8.0) 
        model_path = os.path.join("modelo_2d", "models_2d.json")
        
        try:
            repo = ModelRepository(model_path)
            classifier = repo.load()
            self.recognizer = Recognizer(self.extractor, classifier)
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el modelo en {model_path}.\nEjecuta train_2d.py primero.")
            self.app.mostrar_menu()
            return

        self.strokes = []
        self.current_stroke = []
        self.is_drawing = False
        self.update_timer = None 
        self.alfabeto = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','ñ','o','p','q','r','s','t','u','v','w','x','y','z']

        self._build_ui()

    def _build_ui(self):
        # Configuración de la cuadrícula principal
        self.grid_rowconfigure(0, weight=0) # Barra superior
        self.grid_rowconfigure(1, weight=1) # Contenido
        self.grid_columnconfigure(0, weight=1)

        # --- BARRA SUPERIOR ---
        top_bar = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="transparent")
        top_bar.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 0))
        
        ctk.CTkButton(top_bar, text="← Menú", width=80, fg_color="#E5E7EB", text_color="black", hover_color="#D1D5DB", 
                      command=self.app.mostrar_menu).pack(side="left")
        
        ctk.CTkLabel(top_bar, text="Reconocedor HMM", font=("Arial", 20, "bold"), text_color="black").pack(side="left", padx=15)
        ctk.CTkLabel(top_bar, text=" Modo 2D ", font=("Arial", 12, "bold"), fg_color="#6C5CE7", text_color="white", corner_radius=10).pack(side="left")

        # Botones de la derecha
        ctk.CTkButton(top_bar, text="↺ Reiniciar captura", width=130, fg_color="#6C5CE7", hover_color="#5a4fcf", command=self.clear_canvas).pack(side="right", padx=5)
        ctk.CTkButton(top_bar, text="🗑 Limpiar lienzo", width=120, fg_color="#EF4444", hover_color="#DC2626", command=self.clear_canvas).pack(side="right", padx=5)
        ctk.CTkButton(top_bar, text="↶ Deshacer trazo", width=120, fg_color="#E5E7EB", text_color="black", hover_color="#D1D5DB", command=self.undo_stroke).pack(side="right", padx=5)

        # --- CONTENEDOR PRINCIPAL ---
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=3) # Lienzo
        content_frame.grid_columnconfigure(1, weight=1) # Panel

        # --- SECCIÓN IZQUIERDA: LIENZO ---
        left_frame = ctk.CTkFrame(content_frame, fg_color="white", corner_radius=15)
        left_frame.grid(row=0, column=0, padx=(0, 15), sticky="nsew")
        
        # Canvas nativo
        self.canvas = tk.Canvas(left_frame, bg="white", highlightthickness=0, cursor="pencil")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Textos de ayuda sobre el lienzo
        self.lbl_placeholder = ctk.CTkLabel(left_frame, text="Dibuja aquí una letra minúscula", font=("Arial", 20), text_color="#A0A0A0")
        self.lbl_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        
        ctk.CTkLabel(left_frame, text="Empieza a dibujar con el ratón o con el dedo", font=("Arial", 12), text_color="gray").pack(side="bottom", pady=10)

        # --- SECCIÓN DERECHA: PANEL ---
        right_frame = ctk.CTkFrame(content_frame, fg_color="white", corner_radius=15)
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Panel de la letra principal
        self.top_letter_frame = ctk.CTkFrame(right_frame, fg_color="#6C5CE7", corner_radius=15)
        self.top_letter_frame.pack(fill="x", padx=15, pady=15)
        
        self.lbl_top_letter = ctk.CTkLabel(self.top_letter_frame, text="?", font=("Arial", 44, "bold"), width=70, height=70, fg_color="white", text_color="#6C5CE7", corner_radius=10)
        self.lbl_top_letter.pack(side="left", padx=15, pady=15)
        
        info_frame = ctk.CTkFrame(self.top_letter_frame, fg_color="transparent")
        info_frame.pack(side="left", padx=5)
        ctk.CTkLabel(info_frame, text="Letra más probable", font=("Arial", 14), text_color="white").pack(anchor="w")
        self.lbl_top_prob = ctk.CTkLabel(info_frame, text="—", font=("Arial", 24, "bold"), text_color="white")
        self.lbl_top_prob.pack(anchor="w")

        ctk.CTkLabel(right_frame, text="P(carácter | O1:t)", font=("Arial", 12, "bold"), text_color="#555555").pack(anchor="w", padx=20, pady=(10, 0))

        self.prob_frame = ctk.CTkScrollableFrame(right_frame, fg_color="transparent")
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # CREAMOS LOS 27 RENGLONES DESDE EL INICIO (Con el 3.7% por defecto)
        self.letter_ui = {}
        prob_inicial = 1.0 / len(self.alfabeto)
        
        for char in self.alfabeto:
            f = ctk.CTkFrame(self.prob_frame, corner_radius=8, fg_color="transparent", height=40)
            f.pack(fill="x", pady=2)
            
            # Caja de la letra (Izquierda)
            lbl_box = ctk.CTkLabel(f, text=char, font=("Arial", 18, "bold"), width=35, height=35, fg_color="#EEF2FF", text_color="#4F46E5", corner_radius=8)
            lbl_box.pack(side="left", padx=10, pady=2)
            
            # Nombre de la letra
            lbl_char = ctk.CTkLabel(f, text=char, font=("Arial", 16, "bold"), text_color="black", width=20)
            lbl_char.pack(side="left", padx=5)
            
            # Porcentaje (Derecha)
            lbl_pct = ctk.CTkLabel(f, text="3.7%", font=("Arial", 12), text_color="black", width=45)
            lbl_pct.pack(side="right", padx=10)
            
            # Barra de progreso (Derecha)
            prog = ctk.CTkProgressBar(f, width=120, height=10, progress_color="#A5B4FC", fg_color="#E5E7EB")
            prog.set(prob_inicial)
            prog.pack(side="right", padx=5)
            
            self.letter_ui[char] = {"frame": f, "box": lbl_box, "char_lbl": lbl_char, "prog": prog, "pct": lbl_pct}

    def start_draw(self, event):
        self.is_drawing = True
        self.lbl_placeholder.place_forget() # Ocultar texto de ayuda
        self.current_stroke = [(event.x, event.y)]
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill="black", outline="")

    def draw(self, event):
        if self.is_drawing:
            x, y = event.x, event.y
            if self.current_stroke:
                px, py = self.current_stroke[-1]
                if abs(x - px) > 2 or abs(y - py) > 2:
                    self.canvas.create_line(px, py, x, y, fill="black", width=4, capstyle=tk.ROUND, smooth=True)
                    self.current_stroke.append((x, y))
            
            if self.update_timer:
                self.after_cancel(self.update_timer)
            self.update_timer = self.after(50, self.update_predictions)

    def stop_draw(self, event):
        if self.is_drawing and len(self.current_stroke) > 1:
            self.strokes.append(self.current_stroke)
        self.is_drawing = False
        self.current_stroke = []
        self.update_predictions()

    def undo_stroke(self):
        self.canvas.delete("all")
        if self.strokes:
            self.strokes.pop()
        for stroke in self.strokes:
            for i in range(len(stroke)-1):
                self.canvas.create_line(stroke[i][0], stroke[i][1], stroke[i+1][0], stroke[i+1][1], fill="black", width=4, smooth=True)
        self.update_predictions()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.strokes = []
        self.current_stroke = []
        self.lbl_top_letter.configure(text="?")
        self.lbl_top_prob.configure(text="—")
        self.lbl_placeholder.place(relx=0.5, rely=0.5, anchor="center")
        
        # Restaurar valores por defecto (3.7%)
        prob_inicial = 1.0 / len(self.alfabeto)
        for char in self.alfabeto:
            ui = self.letter_ui[char]
            ui["prog"].set(prob_inicial)
            ui["pct"].configure(text="3.7%")
            ui["frame"].pack(fill="x", pady=2) # Reordenar al alfabeto por defecto
        
    def update_predictions(self):
        all_strokes = self.strokes.copy()
        if self.current_stroke:
            all_strokes.append(self.current_stroke)

        total_puntos = sum(len(s) for s in all_strokes)
        if total_puntos < 5:
            return

        posteriors = self.recognizer.posteriors(all_strokes)
        if not posteriors:
            return

        # 1. Actualizar letra principal
        best_char, best_prob = posteriors[0]
        self.lbl_top_letter.configure(text=best_char)
        self.lbl_top_prob.configure(text=f"{best_prob*100:.1f}%")

        # 2. Ocultar todos los frames
        for char in self.alfabeto:
            self.letter_ui[char]["frame"].pack_forget()

        # 3. Volver a mostrarlos ordenados
        for i, (char, prob) in enumerate(posteriors):
            if char not in self.letter_ui:
                continue
                
            ui = self.letter_ui[char]
            ui["frame"].pack(fill="x", pady=2) # Se empaqueta en el nuevo orden
            ui["pct"].configure(text=f"{prob*100:.1f}%")
            ui["prog"].set(prob)
            
            # Resaltar la ganadora
            if i == 0:
                ui["box"].configure(fg_color="#6C5CE7", text_color="white")
                ui["char_lbl"].configure(text_color="#6C5CE7")
                ui["prog"].configure(progress_color="#6C5CE7")
            else:
                ui["box"].configure(fg_color="#EEF2FF", text_color="#4F46E5")
                ui["char_lbl"].configure(text_color="black")
                ui["prog"].configure(progress_color="#A5B4FC")

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
                    self.canvas.create_line(x1, y1, x2, y2, fill="#00ffcc", width=4, dash=(6, 4), tags="projection")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reconocedor HMM - Procesos Estocásticos")
        self.geometry("1150x750")
        self.minsize(1000, 650)
        
        self.container = ctk.CTkFrame(self, fg_color="#F3F4F6") # Fondo gris muy claro
        self.container.pack(fill="both", expand=True)
        
        self.menu_view = VentanaMenu(self.container, self)
        self.canvas_view = VentanaCanvas2D(self.container, self)
        
        self.mostrar_menu()

    def mostrar_menu(self):
        self.canvas_view.pack_forget()
        self.menu_view.pack(fill="both", expand=True)

    def mostrar_canvas(self):
        self.menu_view.pack_forget()
        self.canvas_view.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()