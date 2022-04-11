import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import *
import sympy as sp

img1 = cv2.imread('imagen_jpg1.jpg')
img2 = cv2.imread('imagen_jpg2.jpg')
img3=img1[1:600,1:300,:]
img4=img2[1:600,1:300,:]

color = ('b','g','r')

cv2.imshow("Imagen 1", img3), cv2.moveWindow("Imagen 1", 0, 0)

img5 = cv2.imread('imagen_jpg1.jpg', cv2.IMREAD_GRAYSCALE)
img5 = cv2.equalizeHist(img5)

for i,col in enumerate(color):
    histimg3 = cv2.calcHist([img3],[i],None,[256],[0,256])
    plt.plot(histimg3,color = col)
    plt.xlim([0,256])
hist1 = cv2.calcHist([img5], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(img3,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histimg3,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(img5,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist1,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

cv2.imshow("Imagen 2", img4), cv2.moveWindow("Imagen 2", 1066, 0)

img6 = cv2.imread('imagen_jpg2.jpg', cv2.IMREAD_GRAYSCALE)
img6 = cv2.equalizeHist(img6)

for i,col in enumerate(color):
    histimg4 = cv2.calcHist([img4],[i],None,[256],[0,256])
    plt.plot(histimg4,color = col)
    plt.xlim([0,256])
hist2 = cv2.calcHist([img6], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(img4,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histimg4,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(img6,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist2,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

suma1 = cv2.addWeighted(img3,1,img4,1,0)

cv2.imshow('suma1',suma1), cv2.moveWindow("suma1", 533, 0)
cv2.imwrite('suma1.jpg',suma1)

suma1h = cv2.imread('suma1.jpg', cv2.IMREAD_GRAYSCALE)
suma1h = cv2.equalizeHist(suma1h)

for i,col in enumerate(color):
    histsuma1 = cv2.calcHist([suma1],[i],None,[256],[0,256])
    plt.plot(histsuma1,color = col)
    plt.xlim([0,256])
hist3 = cv2.calcHist([suma1h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(suma1,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histsuma1,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(suma1h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist3,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()


G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('suma1')
elif G == 27:  
 	cv2.destroyAllWindows()

suma2 = cv2.add(img3,img4)

cv2.imshow('suma2',suma2), cv2.moveWindow("suma2", 533, 0)
cv2.imwrite('suma2.jpg',suma2)

suma2h = cv2.imread('suma2.jpg', cv2.IMREAD_GRAYSCALE)
suma2h = cv2.equalizeHist(suma2h)

for i,col in enumerate(color):
    histsuma2 = cv2.calcHist([suma2],[i],None,[256],[0,256])
    plt.plot(histsuma2,color = col)
    plt.xlim([0,256])
hist4 = cv2.calcHist([suma2h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(suma2,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histsuma2,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(suma2h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist4,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('suma2')
elif G == 27:  
    cv2.destroyAllWindows()

suma3 = img3 + img4

cv2.imshow('suma3',suma3), cv2.moveWindow("suma3", 533, 0)
cv2.imwrite('suma3.jpg',suma3)

suma3h = cv2.imread('suma3.jpg', cv2.IMREAD_GRAYSCALE)
suma3h = cv2.equalizeHist(suma3h)

for i,col in enumerate(color):
    histsuma3 = cv2.calcHist([suma3],[i],None,[256],[0,256])
    plt.plot(histsuma3,color = col)
    plt.xlim([0,256])
hist5 = cv2.calcHist([suma3h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(suma3,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histsuma3,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(suma3h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist5,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('suma3')
elif G == 27:  
    cv2.destroyAllWindows()

resta1 = cv2.subtract(img3,img4)

cv2.imshow('resta1',resta1), cv2.moveWindow("resta1", 533, 0)
cv2.imwrite('resta1.jpg',resta1)

resta1h = cv2.imread('resta1.jpg', cv2.IMREAD_GRAYSCALE)
resta1h = cv2.equalizeHist(resta1h)

for i,col in enumerate(color):
    histresta1 = cv2.calcHist([resta1],[i],None,[256],[0,256])
    plt.plot(histresta1,color = col)
    plt.xlim([0,256])
hist6 = cv2.calcHist([resta1h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(resta1,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histresta1,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(resta1h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist6,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('resta1')
elif G == 27:  
    cv2.destroyAllWindows()


resta2 = cv2.absdiff(img3,img4)

cv2.imshow('resta2',resta2), cv2.moveWindow("resta2", 533,0)
cv2.imwrite('resta2.jpg',resta2)

resta2h = cv2.imread('resta2.jpg', cv2.IMREAD_GRAYSCALE)
resta2h = cv2.equalizeHist(resta2h)

for i,col in enumerate(color):
    histresta2 = cv2.calcHist([resta2],[i],None,[256],[0,256])
    plt.plot(histresta2,color = col)
    plt.xlim([0,256])
hist7 = cv2.calcHist([resta2h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(resta2,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histresta2,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(resta2h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist7,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('resta2')
elif G == 27:  
    cv2.destroyAllWindows()

resta3 = img3 - img4

cv2.imshow('resta3',resta3), cv2.moveWindow("resta3", 533, 0)
cv2.imwrite('resta3.jpg',resta3)

resta3h = cv2.imread('resta3.jpg', cv2.IMREAD_GRAYSCALE)
resta3h = cv2.equalizeHist(resta3h)

for i,col in enumerate(color):
    histresta3 = cv2.calcHist([resta3],[i],None,[256],[0,256])
    plt.plot(histresta3,color = col)
    plt.xlim([0,256])
hist8 = cv2.calcHist([resta3h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(resta3,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histresta3,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(resta3h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist8,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('resta3')
elif G == 27:  
    cv2.destroyAllWindows()

mult1 = cv2.multiply(img3,img4)

cv2.imshow('multiplicacion1',mult1), cv2.moveWindow("multiplicacion1", 533, 0)
cv2.imwrite('multiplicacion1.jpg',mult1)

mult1h = cv2.imread('multiplicacion1.jpg', cv2.IMREAD_GRAYSCALE)
mult1h = cv2.equalizeHist(mult1h)

for i,col in enumerate(color):
    histmult1 = cv2.calcHist([mult1],[i],None,[256],[0,256])
    plt.plot(histmult1,color = col)
    plt.xlim([0,256])
hist9 = cv2.calcHist([mult1h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(mult1,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histmult1,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(mult1h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist9,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('multiplicacion1')
elif G == 27:  
    cv2.destroyAllWindows()

mult2 = img3 * img4

cv2.imshow('multiplicacion2',mult2), cv2.moveWindow("multiplicacion2", 533, 0)
cv2.imwrite('multiplicacion2.jpg',mult2)

mult2h = cv2.imread('multiplicacion2.jpg', cv2.IMREAD_GRAYSCALE)
mult2h = cv2.equalizeHist(mult2h)

for i,col in enumerate(color):
    histmult2 = cv2.calcHist([mult2],[i],None,[256],[0,256])
    plt.plot(histmult2,color = col)
    plt.xlim([0,256])
hist10 = cv2.calcHist([mult2h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(mult2,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histmult2,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(mult2h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist10,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('multiplicacion2')
elif G == 27:  
    cv2.destroyAllWindows()

div1 = cv2.divide(img3,img4)

cv2.imshow('division1',div1), cv2.moveWindow("division1", 533, 0)
cv2.imwrite('division1.jpg',div1)

div1h = cv2.imread('division1.jpg', cv2.IMREAD_GRAYSCALE)
div1h = cv2.equalizeHist(div1h)

for i,col in enumerate(color):
    histdiv1 = cv2.calcHist([div1],[i],None,[256],[0,256])
    plt.plot(histdiv1,color = col)
    plt.xlim([0,256])
hist11 = cv2.calcHist([div1h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(div1,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histdiv1,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(div1h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist11,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('division1')
elif G == 27:  
    cv2.destroyAllWindows()



c = 255 / np.log(1 + np.max(img3)) 
log_img3 = c * (np.log(img3 + 1))    
log_img3 = np.array(log_img3, dtype = np.uint8) 
c = 255 / np.log(1 + np.max(img4)) 
log_img4 = c * (np.log(img4 + 1))    
log_img4 = np.array(log_img4, dtype = np.uint8)
logn1 = cv2.addWeighted(log_img3,1,log_img4,1,0)

cv2.imshow('logaritmo1',logn1), cv2.moveWindow("logaritmo1", 533, 0)
cv2.imwrite('logaritmo1.jpg',logn1)

logn1h = cv2.imread('logaritmo1.jpg', cv2.IMREAD_GRAYSCALE)
logn1h = cv2.equalizeHist(logn1h)

for i,col in enumerate(color):
    histlogn1 = cv2.calcHist([logn1],[i],None,[256],[0,256])
    plt.plot(histlogn1,color = col)
    plt.xlim([0,256])
hist13 = cv2.calcHist([logn1h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(logn1,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histlogn1,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(logn1h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist13,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('logaritmo1')
elif G == 27:  
    cv2.destroyAllWindows()

potencia1 = cv2.pow(img3, 2)
potencia2 = cv2.pow(img4, 2)
potenciaT1 = cv2.addWeighted(potencia1,1,potencia2,1,0)

cv2.imshow('potencia1',potenciaT1), cv2.moveWindow("potencia1", 533, 0)
cv2.imwrite('potencia1.jpg',potenciaT1)

potencia1h = cv2.imread('potencia1.jpg', cv2.IMREAD_GRAYSCALE)
potencia1h = cv2.equalizeHist(potencia1h)

for i,col in enumerate(color):
    histpotencia1 = cv2.calcHist([potencia1],[i],None,[256],[0,256])
    plt.plot(histpotencia1,color = col)
    plt.xlim([0,256])
hist15 = cv2.calcHist([potencia1h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(potenciaT1,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histpotencia1,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(potencia1h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist15,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('potencia1')
elif G == 27:  
    cv2.destroyAllWindows()

potencia3 = img3**2
potencia4 = img4**2
potenciaT2 = cv2.addWeighted(potencia3,1,potencia4,1,0)

cv2.imshow('potencia2',potenciaT2), cv2.moveWindow("potencia2", 533, 0)
cv2.imwrite('potencia2.jpg',potenciaT2)

potencia2h = cv2.imread('potencia2.jpg', cv2.IMREAD_GRAYSCALE)
potencia2h = cv2.equalizeHist(potencia2h)

for i,col in enumerate(color):
    histpotencia2 = cv2.calcHist([potencia2],[i],None,[256],[0,256])
    plt.plot(histpotencia2,color = col)
    plt.xlim([0,256])
hist16 = cv2.calcHist([potencia2h], [0], None, [256], [0, 256])

fig, ax=plt.subplots(2,2)
ax[0,0].imshow(potenciaT2,cmap='gray')
ax[0,0].set_title('Imagen 1')
ax[0,0].axis('off')

ax[0,1].plot(histpotencia2,color='gray')
ax[0,1].set_title('Histograma 1')

ax[1,0].imshow(potencia2h,cmap='gray')
ax[1,0].set_title('Imagen 2')
ax[1,0].axis('off')

ax[1,1].plot(hist16,color='gray')
ax[1,1].set_title('Histograma 2')

plt.show()

G = cv2.waitKey(0) 
if G == ord('g'):
    cv2.destroyWindow('potencia2')
elif G == 27:  
    cv2.destroyAllWindows()



cv2.waitKey(0) 
cv2.destroyAllWindows() 