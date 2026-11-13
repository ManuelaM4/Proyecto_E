import json
import os
import numpy as np

def extraer_puntos_planos(elemento):
    """
    Recorre recursivamente una estructura anidada y devuelve 
    una lista plana de todos los números encontrados.
    """
    puntos = []
    if isinstance(elemento, (int, float)):
        puntos.append(elemento)
    elif isinstance(elemento, (list, tuple)):
        for item in elemento:
            puntos.extend(extraer_puntos_planos(item))
    return puntos

def main():
    ruta_2d = "modelo_2d/glifo-dataset-2026-11-12.json"
    if not os.path.exists(ruta_2d):
        print(f"❌ No se encontró {ruta_2d}")
        return
        
    with open(ruta_2d, 'r', encoding='utf-8') as f:
        data_2d = json.load(f)
    
    print(f"✅ Cargado dataset 2D con {len(data_2d)} letras.")
    
    data_3d = {}
    
    for letra, muestras in data_2d.items():
        data_3d[letra] = []
        
        # Si 'muestras' no es una lista de trazos, lo envolvemos
        if not isinstance(muestras, list):
            continue
            
        for muestra in muestras:
            # 1. Aplanar el trazo para obtener una lista pura de números [x1, y1, p1, x2, y2, p2, ...]
            numeros = extraer_puntos_planos(muestra)
            
            if len(numeros) < 4: # Necesitamos al menos 2 puntos (x,y)
                continue
                
            trazo_3d = []
            # 2. Iterar en saltos de 3 (asumiendo que cada punto tiene x, y, presion)
            # Si tu dataset tiene solo x, y, cambia el rango a step=2
            for i in range(0, len(numeros) - 2, 3):
                x, y = numeros[i], numeros[i+1]
                
                # Normalizar a [-1, 1] asumiendo lienzo de 1000x1000
                x_norm = (x / 500.0) - 1.0
                y_norm = (y / 500.0) - 1.0
                
                # Ajustar al círculo unitario
                r2 = x_norm**2 + y_norm**2
                if r2 > 1.0:
                    r = np.sqrt(r2)
                    x_norm /= r
                    y_norm /= r
                    r2 = 1.0
                
                z = np.sqrt(max(0, 1 - r2))
                trazo_3d.append([float(x_norm), float(y_norm), float(z)])
            
            if len(trazo_3d) > 2:
                data_3d[letra].append(trazo_3d)
    
    os.makedirs("modelo_3d", exist_ok=True)
    ruta_3d = "modelo_3d/glifo-3d.json"
    with open(ruta_3d, 'w', encoding="utf-8") as f:
        json.dump(data_3d, f)
    
    print(f"✅ Dataset 3D generado exitosamente en: {ruta_3d}")

if __name__ == "__main__":
    main()