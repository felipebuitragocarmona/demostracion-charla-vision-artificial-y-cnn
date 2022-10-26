import cv2
import numpy as np

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
    tamañoKernel = 15
    kernel = np.ones((tamañoKernel, tamañoKernel), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    #Detección Figura
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areaMin = 1000
    #Análisis de cada una de las figuras
    for figuraActual in figuras:
        areaFigura=cv2.contourArea(figuraActual)
        if areaFigura>=areaMin:
            #Cálculo de vértices
            vertices = cv2.approxPolyDP(figuraActual, 0.05 * cv2.arcLength(figuraActual, True), True)
            #Análisis de cantidad de vértices
            if len(vertices)==3:
                imagen = escribirMensaje("Triángulo", imagen, figuraActual)
            elif len(vertices)==4:
                imagen = escribirMensaje("Cuadrado", imagen, figuraActual)
            elif len(vertices)==5:
                imagen=escribirMensaje("Pentágono",imagen,figuraActual)
    return imagen
def escribirMensaje(mensaje,imagen,figuraActual):
    cv2.putText(imagen, mensaje, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.drawContours(imagen, [figuraActual], 0, (0, 0, 255), 2)
    return imagen

#Apertura cámara
video=cv2.VideoCapture(0)
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





