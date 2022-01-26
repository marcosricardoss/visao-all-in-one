import cv2 as cv
import numpy as np
import math
import imutils
from app.visao.preprocessing.filter import *

'''
Does: find the two pcbs and crop them in to two different images
Arguments: image
'''
def segment_pcbs(image, screw_cascade):
    # resize
    if (image.shape[1] != 1920):
        image = imutils.resize(image, width=1920)

    # grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)  #<-- Preciso para o cascade
    
    screws = screw_cascade.detectMultiScale(gray, minNeighbors = 3)

    # Erro quando não encontra nenhum screw
    # ERRO: AttributeError: 'tuple' object has no attribute 'shape'
    try: 
        if (screws.shape[0] < 4):
            print("Screws < 4")
            pcbs = None
        elif screws.shape[0] >= 4:
            print("Screws >= 4")
            screws = filter_screws(gray, screws)
            if screws == None:
                print("Algo deu errado com os parafusos!\n")
                pcbs = None
    except AttributeError:
        print("Algo deu errado com os parafusos!\n")
        pcbs = None

    cx = 0.0
    cy = 0.0
    # Procurando centro de rotação
    for(x, y) in screws:
        # cv.circle(draw_screws, (int(x), int(y)), 10, (255,255,0), -1)
        cx = cx + x
        cy = cy + y
    center = (int(cx//4), int(cy//4))
    
   # Ordenando os screws para dimensionar a imagem
    scrD = {}
    for(x, y) in screws:
        if (x < center[0]):
            if (y < center[1]):
                scrD["lt"] = (x, y)
            else:
                scrD["lb"] = (x, y)
        else: 
            if (y < center[1]):
                scrD["rt"] = (x, y)
            else:
                scrD["rb"] = (x, y)

    # Translação para o centro da imagem
    image_h, image_w = image.shape[0], image.shape[1]
    cix = image_w//2
    ciy = image_h//2
    tx = cix - center[0]
    ty = ciy - center[1]
    M = np.float32([ [1,0,tx], [0,1,ty] ])
    image = cv.warpAffine(image, M, (image_w, image_h))

    try:
        ang = np.degrees(np.arctan2(scrD["lb"][1] - scrD["rt"][1], scrD["rt"][0] - scrD["lb"][0]))
    except:
        print("Algo deu errado com os parafusos!\n")
        return None, None
        
    ang -= 2.61 # correção do ângulo dos parafusos
    # ang *= -1
    ang -= 180

    rows,cols = image.shape[0], image.shape[1]

    # Rotação
    image = imutils.rotate_bound(image, ang)

    # distância em pixels entre os screws
    c1 = np.sqrt((scrD["lb"][0] - scrD["rb"][0])**2 + (scrD["lb"][1] - scrD["rb"][1])**2)
    c2 = np.sqrt((scrD["lt"][0] - scrD["rt"][0])**2 + (scrD["lt"][1] - scrD["rt"][1])**2)
    c = (c1 + c2)/2

    #pxm = c/182  #pixels por milímetro
    pxm = c/163  #pixels por milímetro (distancia entre os parafusos)
    pcbx = 160 * pxm # largura da pcb em pixels (140 mm)
    pcby = 115 * pxm # altura da pcb em pixels
    deslocamento_left = 25 * pxm # folga horizaontal
    deslocamento_right = 20 * pxm # folga horizaontal

    image_h, image_w = image.shape[0], image.shape[1]
    cix = image_w//2
    ciy = image_h//2
    
    left = image[int(ciy-pcby):int(ciy), int(cix-pcbx+deslocamento_left):int(cix+deslocamento_left),:]
    right = image[int(ciy):int(ciy+pcby), int(cix-deslocamento_right):int(cix+pcbx-deslocamento_right),:]

    return left, right