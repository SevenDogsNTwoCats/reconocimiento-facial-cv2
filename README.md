# Reconocimiento Facial con cv2
Este proyecto utiliza la biblioteca OpenCV (cv2) para realizar el reconocimiento facial.

## Estructura del Proyecto
El proyecto consta de varios archivos y directorios:

- capturarRostroCamara.py: Este script captura rostros a través de la cámara.
- capturarRostroVideo.py: Este script captura rostros a través de un video.
entrenar.py: Este script se utiliza para entrenar el modelo de reconocimiento facial.
- reconocimientoFacial.py: Este script realiza el reconocimiento facial.
data/: Este directorio contiene los datos utilizados para el entrenamiento y la detección.
- haarcascade_frontalface_default.xml: Este archivo XML contiene la cascada Haar utilizada para la detección de rostros.
## Cómo Usar
Ejecute capturarRostroCamara.py o capturarRostroVideo.py para capturar rostros.
Ejecute entrenar.py para entrenar el modelo de reconocimiento facial.
Ejecute reconocimientoFacial.py para realizar el reconocimiento facial.


## Dependencias
- OpenCV
- imutils
- os
- numpy

## Autor
Alhanis Espinal - alhanis.espinal@outlook.com

## Fecha
2024/05/09
