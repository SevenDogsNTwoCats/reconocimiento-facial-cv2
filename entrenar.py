'''
Codigo para entrenar un modelo de reconocimiento de rostros, con las imagenes guardadas en la carpeta data.

Autor: alhanis.espinal@outlook.com
Fecha: 9-5-2024
'''

import cv2 as cv
import os
import numpy as np
from time import time

# pasar la carpeta
# os.getcwd() devuelve la ruta actual del proyecto
ruta = os.getcwd()
# carpeta donde se guardan los modelos
rutaCompleta = os.path.join(ruta, 'data')
# lista de modelos
listaModelos = os.listdir(rutaCompleta)
# print('Modelos encontrados: ', listaModelos)

ids = []
rostrosData = []
id = 0
tiempoInicial = time()
# recorrer los modelos
for modelos in listaModelos:
    # ruta de el modelo
    rutaCompletaImagenes = os.path.join(rutaCompleta, modelos)
    # leer las imagenes de el modelo
    for archivo in os.listdir(rutaCompletaImagenes):
        print('Rostros: ', modelos + '/' + archivo)
        # agregar a la lista de ids y las propiedades de la imagen
        ids.append(id)
        # 0 es para que la imagen se lea en escala de grises
        rostrosData.append(cv.imread(os.path.join(rutaCompletaImagenes, archivo), 0))
        imagenes = cv.imread(os.path.join(rutaCompletaImagenes, archivo), 0)    
    # incrementar el id para cada modelo
    id += 1
    tiempoLectura = time() - tiempoInicial
    print('Tiempo de lectura: ', tiempoLectura)

# proceso de entrenamiento
# crear el modelo de entrenamiento
# modelo de entrenamiento de reconocimiento de rostros 1
entrenamientoEigenRecongnizer = cv.face.EigenFaceRecognizer_create()
print('Entrenando...')
# entrenar el modelo con las propiedades de las imagenes
entrenamientoEigenRecongnizer.train(rostrosData, np.array(ids))
print ('Tiempo de entrenamiento: ', time() - tiempoLectura)
print('Modelo entrenado correctamente, guardando modelo...')
# guardar el modelo entrenado
entrenamientoEigenRecongnizer.write('EntrenamientoEigenFaceRecongnizer.xml')
print('Modelo guardado correctamente!')