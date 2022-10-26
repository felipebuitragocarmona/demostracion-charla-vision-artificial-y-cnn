import cv2

from CNN.Prediccion import Prediccion

clases=["numero 0","numero 1","numero 2","numero 3","numero 4","numero 5","numero 6","numero 7","numero 8","numero 9"]

width=28
heigth=28

miModeloCNN=Prediccion("models/modeloA.h5",width,heigth)
imagen_seleccionada=cv2.imread("dataset/test/9/9_4.jpg")
imagen_seleccionada=cv2.cvtColor(imagen_seleccionada,cv2.COLOR_BGR2GRAY)

clasePredicha=miModeloCNN.predecir(imagen_seleccionada)
print(clasePredicha)
print("La imagen cargada corresponde al ",clases[clasePredicha])
while True:
    cv2.imshow("imagen",imagen_seleccionada)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()