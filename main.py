"""
main.py
Punto de entrada.
"""
from GUI import App2D

def main():
    # CustomTkinter ya crea su propia ventana raíz internamente
    app = App2D()
    app.mainloop()

if __name__ == "__main__":
    main()