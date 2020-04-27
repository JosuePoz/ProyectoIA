# -*- coding: utf-8 -*-
"""
@author: kurotensei
"""

import cv2
import numpy as np

try:
    imgo = cv2.imread("foto3.png")
    img = cv2.resize(imgo, (500,500))
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suavizado=cv2.GaussianBlur(img,(7,7),0)
    bordeado = cv2.Canny(suavizado,10,350)
    _,th = cv2.threshold(gris,100,255,cv2.THRESH_BINARY)

    imagen, contorno, j = cv2.findContours(bordeado, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contorno, -1,(0,0,255),3)
    
    cv2.imshow('suavizada',suavizado)
    cv2.imshow('umbral',th)
    cv2.imshow('imagen',img)
    cv2.waitKey(0)
    
except Exception as e:
    print(str(e))







