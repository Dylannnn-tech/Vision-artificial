import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk

# Modelo YOLO
model = YOLO("yolov8n.pt")

cap = None
running = False

# Carpeta
carpeta = "capturas_personas"
os.makedirs(carpeta, exist_ok=True)

cooldown = 30
contador = 0

def iniciar():
    global cap, running
    cap = cv2.VideoCapture(0)
    running = True
    actualizar()

def detener():
    global running, cap
    running = False
    if cap:
        cap.release()

def actualizar():
    global contador

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    resultados = model(frame)[0]

    persona_detectada = False

    for box in resultados.boxes:
        clase_id = int(box.cls[0])
        nombre = model.names[clase_id]

        if nombre == "person":
            persona_detectada = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    if persona_detectada:
        estado_label.config(text="PERSONA DETECTADA", fg="red")

        if contador == 0:
            nombre = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            ruta = os.path.join(carpeta, nombre)
            cv2.imwrite(ruta, frame)
            contador = cooldown
    else:
        estado_label.config(text="Sin detección", fg="green")

    if contador > 0:
        contador -= 1

    # Convertir imagen para Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    panel.imgtk = imgtk
    panel.config(image=imgtk)

    root.after(10, actualizar)

# UI
root = tk.Tk()
root.title("AI Security Camera")

panel = tk.Label(root)
panel.pack()

estado_label = tk.Label(root, text="Sin detección", font=("Arial", 16))
estado_label.pack()

btn_iniciar = tk.Button(root, text="Iniciar", command=iniciar)
btn_iniciar.pack(side="left", padx=10, pady=10)

btn_detener = tk.Button(root, text="Detener", command=detener)
btn_detener.pack(side="right", padx=10, pady=10)

root.mainloop()
