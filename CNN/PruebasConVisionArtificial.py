import cv2

from CNN.Prediccion import Prediccion
import cv2
import numpy as np

clases=["numero 0","numero 1","numero 2","numero 3","numero 4","numero 5","numero 6","numero 7","numero 8","numero 9"]
width=28
heigth=28
miModeloCNN=Prediccion("models/modeloA.h5",width,heigth)


nameWindow="Calculadora"
def nothing(x):
    pass
def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,0,255,nothing)
    cv2.createTrackbar("max", nameWindow, 100, 255, nothing)
    cv2.createTrackbar("kernel", nameWindow, 1, 100, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 500, 10000, nothing)
def calcularAreas(figuras):
    areas=[]
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas
def detectarForma(imagen):
    #Conversión imagen a escala grises
    imagenGris=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    #Parámetros calculados a partir del estado del entorno
    min = cv2.getTrackbarPos("min", nameWindow)
    max = cv2.getTrackbarPos("max", nameWindow)
    # Detección de bordes
    bordes=cv2.Canny(imagenGris,min,max)
    #Operación morfológica
    tamañoKernel = cv2.getTrackbarPos("kernel", nameWindow)
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    #Detección Figura
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areaMin = cv2.getTrackbarPos("areaMin", nameWindow)

    #Análisis de cada una de las figuras
    i=0
    for figuraActual in figuras:
        areaFigura=cv2.contourArea(figuraActual)
        if areaFigura>=areaMin and jerarquia[0][i][3]!=-1:
            print(jerarquia[0][i][3])
            #Cálculo de vértices
            vertices = cv2.approxPolyDP(figuraActual, 0.05 * cv2.arcLength(figuraActual, True), True)
            if len(vertices)==4:
                recortar(imagen,figuraActual)

                imagen_seleccionada = cv2.imread("recorte.jpg")
                cv2.imshow("Recorte p", imagen_seleccionada)

                imagen_seleccionada = cv2.cvtColor(imagen_seleccionada, cv2.COLOR_BGR2GRAY)
                #imagen_seleccionada_invertida = cv2.bitwise_not(imagen_seleccionada)
                ret,imagen_seleccionada_invertida=cv2.threshold(imagen_seleccionada, 120, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow("Recorte2", imagen_seleccionada_invertida)

                clasePredicha = miModeloCNN.predecir(imagen_seleccionada_invertida)
                print("La clase es ",clasePredicha)
                print("La imagen cargada corresponde al ",clases[clasePredicha])
                imagen = escribirMensaje("Valor= "+str(clases[clasePredicha]), imagen, figuraActual,90)

                imagen = escribirMensaje("Cuadrado " , imagen, figuraActual,40)
        i=i+1
    return imagen
def recortar(imagen,contorno):
    try:
        x, y, w, h = cv2.boundingRect(contorno)
        recorte = imagen[y+10:y + h-10, x+10:x + w-10]
        cv2.imwrite("recorte.jpg", recorte)
    except:
        pass
    return recorte
def escribirMensaje(mensaje,imagen,figuraActual,y):
    cv2.putText(imagen, mensaje, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.drawContours(imagen, [figuraActual], 0, (0, 0, 255), 2)
    return imagen

#Apertura cámara
video=cv2.VideoCapture(1)
bandera=True
constructorVentana()
while bandera:
    _,imagen=video.read()
    imagen=detectarForma(imagen)
    cv2.imshow("Imagen",imagen)

    #Parar el programa
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        bandera=False
video.release()
cv2.destroyAllWindows()


