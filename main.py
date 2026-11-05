"""
main.py
Punto de entrada.
"""
from GUI import App

def main():
    # CustomTkinter ya crea su propia ventana raíz internamente
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()