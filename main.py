"""
main.py
Punto de entrada.
"""
import tkinter as tk
from GUI import App2D

def main():
    root = tk.Tk()
    app = App2D(root)
    root.mainloop()

if __name__ == "__main__":
    main()