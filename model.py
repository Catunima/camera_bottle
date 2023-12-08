import torch
import cv2
import numpy as np

# Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=r'D:\Projects\Artificial_vision\camara_detector_robot\best.pt')

# Realizo Videocaptura
cap = cv2.VideoCapture(0)

# Inicializamos el contador
contador_botellas = 0

# Inicializamos un conjunto para llevar un registro de las botellas ya contadas
botellas_contadas = set()

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

        # Verificamos si esta botella ya se ha contado
        if botella_id not in botellas_contadas:
            # Incrementamos el contador
            contador_botellas += 1

            # Agregamos esta botella al conjunto de botellas contadas
            botellas_contadas.add(botella_id)

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
