import time
from redis import Redis
from celery.utils.log import get_task_logger
from app import create_celery_app

# Detection imports
import cv2 as cv
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
from app.visao.yolov3.utils import detect_image
from app.visao.yolov3.configs import *
from app.visao.yolov3.yolov4 import Create_Yolo
from app.visao.preprocessing.segment_pcbs import *
from threading import Thread
from datetime import datetime
from gpiozero import Button
import argparse

celery = create_celery_app()
logger = get_task_logger(__name__)
r = Redis(host='all-in-one-redis', port=6379, db=0, decode_responses=True)

def makeDetection(frame, yolo, class_models):
    # Inicializa array
    components = np.ones((6, 2))

    # Separa as duas PCBs
    pcb_left, pcb_right = segment_pcbs(frame)

    try:
        if pcb_left == None and pcb_right == None:
            return None, None, components
    except ValueError:
        pass

    def draw_bboxes(index, image, bboxes):
        # Pegar bboxes azul_roxos
        azul_roxos = []
        for bbox in bboxes:
            class_ind = int(bbox[5])
            if class_ind in range(2):
                azul_roxos.append(bbox)

        # Ordenar azul_roxos por Y
        def takeY(elem):
            return elem[1]

        azul_roxos.sort(key=takeY)

        # Pegar resto dos bboxes
        for bbox in bboxes:
            class_ind = int(bbox[5])
            if class_ind in range(2, 5):
                azul_roxos.append(bbox)

        for bbox in azul_roxos:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = "{:.2f}".format(bbox[4])
            class_ind = int(bbox[5])
            correct = 1

            # Cores dos retangulos de cada classe
            rectangle_colors = {
                '3': (0, 0, 255), # Componente incorreto
                '0': (0, 255, 0), # Componente correto
            }

            # Cores dos scores de cada classe
            text_colors = {
                '3': (0, 0, 0),
                '0': (0, 0, 0),
            }

            placa = {
                'left': 0,
                'right': 1,
            }

            # Coordenadas do bounding box
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
            
            # Classificar se está correto ou incorreto
            prediction = np.array([[1.]])
            if class_ind in range(3):
                component = image[y1-5:y2+5,x1-5:x2+5,:]
                component = cv.resize(component, (32, 32))
                component = cv.cvtColor(component, cv.COLOR_BGR2GRAY)
                prediction = class_models[str(class_ind)](component[np.newaxis,...,np.newaxis])

            image_y = image.shape[0]
            if class_ind in range(2):
                if  (y1+y2)/2 < image_y/3.2:
                    # AZUL 1
                    if class_ind == 1:
                        correct = 3
                    else:
                        correct = 0
                    components[0][placa[index]] = correct
                elif (y1+y2)/2 < image_y/2:
                    # ROXO 1
                    if class_ind == 0:
                        correct = 3
                    else:
                        correct = 0
                    components[1][placa[index]] = correct
                else:
                    # ROXO 2
                    if class_ind == 0:
                        correct = 3
                    else:
                        correct = 0
                    components[2][placa[index]] = correct
            
            # Componente está incorreto
<<<<<<< HEAD
            if prediction[0][0]<0.9:
                correct = False
=======
            else:
                if prediction[0][0] < 0.5:
                    correct = 3
                else:
                    correct = 0
                components[class_ind+1][placa[index]] = correct
>>>>>>> 94f16a0db08f3c1485f562c998826006e3deaba8

            # Desenhar retângulo e score
            bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            cv.rectangle(image, (x1, y1), (x2, y2), rectangle_colors[str(correct)], bbox_thick)

            # get text size
            (text_width, text_height), baseline = cv.getTextSize(score, cv.FONT_HERSHEY_COMPLEX_SMALL,
                                                                fontScale, thickness=1)
            # put filled text rectangle
            cv.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), 
                                            rectangle_colors[str(correct)], thickness=cv.FILLED)
            # put text above rectangle
            cv.putText(image, score, (x1, y1-4), cv.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, text_colors[str(correct)], 1, lineType=cv.LINE_AA)
    
    def detect(index, image):
        bboxes = detect_image(yolo, image, input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES,
                                score_threshold=0.5, iou_threshold=0.3)
        draw_bboxes(index, image, bboxes)

    thread_1 = Thread(target=detect,args=['left', pcb_left])

    thread_1.start()
    detect('right', pcb_right)

    thread_1.join()

    # np.array to list
    components = list(components)

    return pcb_right,pcb_left,components

@celery.task(bind=True)
def long_task(self):
    # Inicialização
    step = 1
    components = np.ones((6, 2))
    self.update_state(state='INITIALIZING', meta={"step":step, "components":components})

    # Carregar a rede neural YOLO
    checkpoints_path = "app/visao/checkpoints/yolov3_C920-13all-50epochs_Tiny"
    yolo = Create_Yolo(input_size=416, CLASSES="app/visao/model_data/classes.txt")
    yolo.load_weights(checkpoints_path)

    # Carregar modelos de classificação 
    class_models = {
        '0': tf.keras.models.load_model('app/visao/classification_models/azul-roxo'),
        '1': tf.keras.models.load_model('app/visao/classification_models/azul-roxo'),
        '2': tf.keras.models.load_model('app/visao/classification_models/pequeno')
    }

    # Carregar pasta das imagens
    DEFAULT_MEDIA_FOLDER = os.environ.get("DEFAULT_MEDIA_FOLDER")

    butaoB = Button(2)
    butaoA = Button(3)

    ind_image = 0
    while True:
        step = 2
        self.update_state(state='READY FOR THE ACTION!', meta={"step":step, "components":components})

        # Inverter e escrever o frame na pasta
        frame = cv.imread(DEFAULT_MEDIA_FOLDER+"opencv_frame_8"+str(ind_image)+".png")
        time.sleep(1)

        # Botão de saída
        if butaoA.is_pressed:
            step = 6
            self.update_state(state='WHY DID YOU LEFT ME?', meta={"step":step, "components":components})
            break

        # Detecção
        elif butaoB.is_pressed:
            step = 3
            self.update_state(state='DETECTION IN PROGRESS...', meta={"step":step, "components":components})

            pcbR,pcbL,components = makeDetection(frame, yolo, class_models)
            ind_image += 1
            if ind_image == 10:
                ind_image = 0
            try:
                if pcbR == None and pcbL == None:
                    step = 5
                    self.update_state(state="PCBS WERE NOT FOUND!", meta={"step":step, "components":components})
                    time.sleep(5)
            except ValueError:
                cv.imwrite(DEFAULT_MEDIA_FOLDER+"left.jpg", pcbL)
                cv.imwrite(DEFAULT_MEDIA_FOLDER+"right.jpg", pcbR)
                step = 4
                self.update_state(state='SHOW TIME!', meta={"step":step, "components":components})
                time.sleep(20)

    return {'status': 'the task have been successfully processed'}

# start the task and send your ID to the frontend via Redis
task_id = r.get('taskid')
if not task_id:
    task = long_task.apply_async()
    r.set('taskid', task.id)