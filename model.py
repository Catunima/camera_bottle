import torch
import cv2
import numpy as np

# Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = r'D:\Projects\Artificial_vision\camara_detector_robot\best.pt')

# Realizo Videocaptura
cap = cv2.VideoCapture(1)

# Empezamos
while True:
    # Realizamos lectura de frames
    ret, frame = cap.read()

    # Correccion de color
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizamos las detecciones
    detect = model(frame)

    info = detect.pandas().xyxy[0]  # im1 predictions
    print(info)

    # Mostramos FPS
    cv2.imshow('Detector de Carros', np.squeeze(detect.render()))

    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()