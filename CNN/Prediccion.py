from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

class Prediccion():
    def __init__(self,ruta_modelo,width, heigth):
        self.ruta_modelo=ruta_modelo
        self.model=load_model(self.ruta_modelo)
        self.width=width
        self.heigth=heigth
    """
    Predice la categoria a la que pertenece una imagen
    
    :param imagen , debe estar a blanco y negro
    """
    def predecir(self,imagen):
        try:
            imagen=cv2.resize(imagen,(self.width,self.heigth))
            imagen = imagen.flatten()
            imagen = imagen / 255
            imagenesCargadas=[]
            imagenesCargadas.append(imagen)
            imagenesCargadasNPA=np.array(imagenesCargadas)
            resultados=self.model.predict(x=imagenesCargadasNPA)
            print("Las probabilidades son=",resultados)
            claseMayor=np.argmax(resultados,axis=1)
            return claseMayor[0]
        except:
            return 0