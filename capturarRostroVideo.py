'''
Codigo es para capturar un rostro de un video y guardarlo en una carpeta.

Autor: alhanis.espinal@outlook.com
Fecha: 9-5-2024
'''

import cv2 as cv
import os
import imutils

# carpeta
nombre = 'elon'
modelo = 'opencv_face_detector_' + nombre

# ruta actual del proyecto
# os.getcwd() devuelve la ruta actual del proyecto
ruta = os.getcwd()

# unir la ruta actual con la carpeta
rutaCompleta = os.path.join(ruta, 'data', modelo)

# si la carpeta no existe
if not os.path.exists(rutaCompleta):
    print('Carpeta no encontrada, se creara la carpeta: ' + modelo)
    # crear la carpeta
    os.makedirs(rutaCompleta)

# ----------------------------reconocimiento de un rostro --------------------------------
# cargar el archivo xml que contiene la informacion para detectar rostros de opencv
ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# capturar video 
camara = cv.VideoCapture('ElonMusk.mp4')
# id de la imagen para luego guardarla
id = 0

while True:
    # leer la captura de la camara
    repuesta, captura = camara.read()
    # si no se puede leer la captura se cierra el programa
    if repuesta == False:
        break
    # redimensionar la captura para que sea mas rapido el proceso de deteccion de rostros y no ocupe tanto espacio
    captura = imutils.resize(captura, width=640)
    # pasar la captura a escala de grises
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    # guardar las propiedades de la captura
    idcaptura = captura.copy()
    # para que sea mas rapido el proceso de deteccion de rostros 1.4 es el 40% de la imagen, 3 es el minimo de vecinos
    cara = ruidos.detectMultiScale(grises, 1.3, 5)
    # dibujar un rectangulo en la cara detectada
    # x,y son las coordenadas de la esquina superior izquierda del rectangulo, w,h son las esquinas del ancho y alto del rectangulo
    for (x, y, w, h) in cara:
        # dibujar un rectangulo en la cara detectada, (0,255,0) es el color del rectangulo, 2 es el grosor del rectangulo
        cv.rectangle(captura, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # almacenar la captura de la cara
        rostroCapturado = idcaptura[y:y+h, x:x+w]
        # tamaño de la imagen del rostro capturado 160x160, cv.INTER_CUBIC es el metodo de interpolacion lo que hace es que la imagen se vea mas nitida
        rostroCapturado = cv.resize(rostroCapturado, (160, 160), interpolation=cv.INTER_CUBIC)
        # modificar cada nombre de la imagen
        cv.imwrite(rutaCompleta + '/rostro_{}.jpg'.format(id), rostroCapturado)
        # aumentar el id para que la imagen no se sobreescriba
        id += 1
    # mostrar la captura de la camara
    cv.imshow('Video rostro', captura)
    # si saca 350 fotos se cierra el programa
    if id == 351:
        break

# liberar la camara y cerrar todas las ventanas
camara.release()
cv.destroyAllWindows()
