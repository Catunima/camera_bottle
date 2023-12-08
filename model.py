import torch
import cv2
import numpy as np
import time

# Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=r'D:\Projects\Artificial_vision\camara_detector_robot\best.pt')

# Realizo Videocaptura
cap = cv2.VideoCapture(0)

# Inicializamos el contador
contador_botellas = 0

# Inicializamos un diccionario para llevar un registro de las botellas detectadas y el tiempo de la última detección
botellas_detectadas = {}

# Definimos el tiempo de espera entre detecciones (en segundos)
tiempo_espera = 8

# Empezamos
while True:
    # Realizamos lectura de frames
    ret, frame = cap.read()

    # Realizamos las detecciones
    detect = model(frame)

    info = detect.pandas().xyxy[0]  # im1 predictions

    # Iteramos sobre las detecciones
    for index, row in info.iterrows():
        botella_id = index  # Puedes usar otra información para identificar la botella si es necesario

        # Verificamos si esta botella ya se ha detectado y si ha pasado suficiente tiempo desde la última detección
        if botella_id not in botellas_detectadas or time.time() - botellas_detectadas[botella_id] > tiempo_espera:
            # Incrementamos el contador
            contador_botellas += 1

            # Actualizamos el tiempo de la última detección para esta botella
            botellas_detectadas[botella_id] = time.time()

    # Dibujamos el número de botellas detectadas en el frame
    cv2.putText(frame, f'Botellas: {contador_botellas}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Dibujamos las cajas delimitadoras de las botellas
    for index, row in info.iterrows():
        x_min, y_min, x_max, y_max = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Mostramos FPS y resultado
    cv2.imshow('Detector de botellas', frame)

    # Leemos el teclado
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
