import tensorflow as tf
import keras
import numpy as np
import cv2
###Importar componentes de la red neuronal
from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten

##############Funciones Requeridas####################
def cargarDatos(fase,numeroCategorias,limite,width,height):
    imagenesCargadas=[]
    valorEsperado=[]
    for categoria in range(0,numeroCategorias):
        for idImagen in range(0,limite[categoria]):
            ruta=fase+str(categoria)+"/"+str(categoria)+"_"+str(idImagen)+".jpg"
            print(ruta)
            imagen=cv2.imread(ruta)
            imagen=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
            imagen = cv2.resize(imagen, (width,height))
            imagen=imagen.flatten()
            imagen=imagen/255
            imagenesCargadas.append(imagen)
            probabilidades=np.zeros(numeroCategorias)
            probabilidades[categoria]=1
            valorEsperado.append(probabilidades)
    imagenesEntrenamiento=np.array(imagenesCargadas)
    valoresEsperados=np.array(valorEsperado)
    return imagenesEntrenamiento,valoresEsperados

#Configuración previa
width=28
height=28
pixeles=width*height
num_chanels=1
img_shape=(width,height,num_chanels)

num_clases=10
cantidadDatosEntrenamiento=[60,60,60,60,60,60,60,60,60,60]
cantidadDatosPruebas=[20,20,20,20,20,20,20,20,20,20]
#Carga de los datos
imagenes,probabilidades=cargarDatos("dataset/train/",num_clases,cantidadDatosEntrenamiento,width,height)

model=Sequential()
#Capa de entrada
model.add(InputLayer(input_shape=(pixeles,)))
model.add(Reshape(img_shape))

#Capas oculta 1
model.add(Conv2D(kernel_size=5,strides=2,filters=14,
                 padding="same",activation="relu",name="capa_1"))

model.add(MaxPool2D(pool_size=2,strides=2))

#Capas oculta 2
model.add(Conv2D(kernel_size=5,strides=2,filters=36,
                 padding="same",activation="relu",name="capa_2"))

model.add(MaxPool2D(pool_size=2,strides=2))

#Aplanamiento  y Capa Densamente Conectada
model.add(Flatten())
model.add(Dense(128,activation="relu"))

#Capa de salida
model.add(Dense(num_clases,activation="softmax"))

#Traducir de keras a tensorflow
model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])
#Entrenamiento
model.fit(x=imagenes,y=probabilidades,epochs=30,batch_size=60)

### Realizar las pruebas
imagenesPrueba,probabilidadesPrueba=cargarDatos("dataset/test/",num_clases,cantidadDatosPruebas,width,height)
resultados=model.evaluate(x=imagenesPrueba,y=probabilidadesPrueba)

print("Accuracy Test=",resultados[1])
print(model.metrics_names)
print(resultados)

### Guardar el modelo
ruta="models/modeloA.h5"
model.save(ruta)

#Resumen - Estructura de la red
model.summary()
