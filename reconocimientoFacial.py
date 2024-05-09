'''
Codigo es para reconocer un rostro de una camara y mostrar el resultado en la captura.

Autor: alhanis.espinal@outlook.com
Fecha: 9-5-2024
'''

import cv2 as cv
import os
import numpy as np
import imutils

# carpeta
# os.getcwd() devuelve la ruta actual del proyecto
ruta = os.getcwd()
# carpeta donde se guardan los modelos
rutaCompleta = os.path.join(ruta, 'data')
# lista de modelos
listaData = os.listdir(rutaCompleta)
# leer el modelo entrenado
entrenamientoEigenRecongnizer = cv.face.EigenFaceRecognizer_create()
entrenamientoEigenRecongnizer.read('EntrenamientoEigenFaceRecongnizer.xml')

# cargar el archivo xml que contiene la informacion para detectar rostros de opencv
ruidos = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# capturar video de la camara
camara = cv.VideoCapture(0)
while True:
    # leer la captura de la camara
    _, captura = camara.read()
    # pasar a escala de grises
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    # guardar las propiedades de la captura
    idcaptura = grises.copy()
    # redimensionar la captura para que sea mas rapido el proceso de deteccion de rostros y no ocupe tanto espacio
    # captura= imutils.resize(captura, width=640)
    # para que sea mas rapido el proceso de deteccion de rostros 1.3 es el 30% de la imagen, 5 es el minimo de vecinos
    cara = ruidos.detectMultiScale(grises, 1.3, 5)
    # dibujar un rectangulo en la cara detectada
    for (x,y,w,h) in cara:
        # almacenar la captura de la cara
        rostroCapturado = idcaptura[y:y+h,x:x+w]
        # tamaño de la imagen del rostro capturado 160x160, cv.INTER_CUBIC es el metodo de interpolacion lo que hace es que la imagen se vea mas nitida
        rostroCapturado = cv.resize(rostroCapturado,(160,160),interpolation=cv.INTER_CUBIC)
        # predecir el resultado
        resultado = entrenamientoEigenRecongnizer.predict(rostroCapturado)
        # mostrar el resultado en la captura
        cv.putText(captura, '{}'.format(resultado), (x,y-5), 1, 1.3, (0,255,0),1,cv.LINE_AA)
        # si el resultado es menor a 5700 es porque es un rostro conocido
        if resultado[1] < 9000:
            # mostrar el nombre de la persona
            cv.putText(captura, '{}'.format(listaData[resultado[0]].split('_')[3]), (x,y-30), 2, 1.1, (0,255,0),1,cv.LINE_AA)
            # dibujar un rectangulo en la cara detectada, (255,0,0) es el color del rectangulo, 2 es el grosor del rectangulo
            cv.rectangle(captura,(x,y),(x+w,y+h),(255,0,0),2)
        else:
            cv.putText(captura, 'Desconocido', (x,y-10), 2, 1.1, (0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura,(x,y),(x+w,y+h),(255,0,0),2)
    # mostrar la captura de la camara
    cv.imshow('Video rostro',captura)
    # si se presiona la tecla q se cierra el programa
    if cv.waitKey(1) == ord('q'):
        break

# liberar la camara y cerrar todas las ventanas
camara.release()
cv.destroyAllWindows()
    

