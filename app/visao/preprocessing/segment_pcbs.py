import cv2 as cv
import numpy as np
import math
import sys
import imutils
from os import listdir

'''
Does: list .png files
Arguments: images path (.png)
Returns: list of images names (without .png)
'''
def list_png_files(path=None):
    if path == None:
        print("Nenhuma pasta foi especificada.")
        return 0

    images = []
    files = [f for f in listdir(path)]
    for f in files:
        if f[len(f)-4:] == ".png":
            images.append(f)

    return images

def closing(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Detecção de bordas
    gray = cv.Canny(gray,100,150)
    # Fechamento
    Kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))
    close = cv.morphologyEx(gray, cv.MORPH_CLOSE, Kernel)

    return close

def fill_holes(mask, size):
    mask = cv.bitwise_not(mask)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        if cv.contourArea(contour) < size:
            cv.fillPoly(mask, pts =[contour], color=(0,0,0))

    mask = cv.bitwise_not(mask)

    return mask

'''
Does: find the two pcbs and crop them in to two different images
Arguments: image
'''
def segment_pcbs(image):
    # resize
    image = imutils.resize(image, width=1920)

    # grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # equalização adaptativa
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # detecção de bordas
    edges = cv.Canny(cl,100,150)

    # encontrar maior linha da imagem (esperamos que seja sempre a horizontal da placa)
    lines = cv.HoughLines(edges,1,np.pi/180,150)
    rho, theta = lines[0][0][0], lines[0][0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    hough = cv.line(edges.copy(),(x1,y1),(x2,y2),150,2)

    # rotacionar imagem -45 a partir dessa linha
    image_rotated = imutils.rotate_bound(image, theta*57.2-45)

    # grayscale e clahe novamente
    gray = cv.cvtColor(image_rotated, cv.COLOR_BGR2GRAY)
    cl = clahe.apply(gray)

    # detecção de bordas
    edges = cv.Canny(cl,100,150)

    # fechamento
    Kernel = cv.getStructuringElement(cv.MORPH_RECT,(21,21))
    close = cv.morphologyEx(edges, cv.MORPH_CLOSE, Kernel)

    # preencher contornos pequenos
    mask = fill_holes(close, 1000)

    # ler template
    template = cv.imread('preprocessing/template-fhd.jpeg',0)

    # resize pra uma proporção boa pra a distância da camera
    template = imutils.resize(template, width=500)
    w, h = template.shape[::-1]

    # specify a threshold
    threshold = 0.4

    # template matching
    res = cv.matchTemplate(mask,template,cv.TM_CCOEFF_NORMED)
    
    if np.all(res < threshold):
        return None, None

    # pegar melhor resultado e recortar
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    x1 = top_left[0]

    # pintar o que já foi pego de preto
    cv.rectangle(mask,top_left, bottom_right, 0, -1)
    
    add_w = 20
    add_h = 120

    pcb1 = image_rotated[top_left[1]-add_w:top_left[1]+w+add_w, top_left[0]-add_h:top_left[0]+h+add_h+20]

    # template matching
    res = cv.matchTemplate(mask,template,cv.TM_CCOEFF_NORMED)
    
    if np.all(res < threshold):
        return None, None

    # pegar melhor resultado e recortar
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    x2 = top_left[0]

    cv.rectangle(mask,top_left, bottom_right, 0, -1)
    
    pcb2 = image_rotated[top_left[1]-add_w:top_left[1]+w+add_w, top_left[0]-add_h:top_left[0]+h+add_h+20]

    # right or left
    if x1 < x2:
        return pcb1, pcb2
    return pcb2, pcb1