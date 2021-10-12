import numpy
import math
import cv2
import imutils
import operator

def show_image(img, image_name = 'Image'):
    newimage = imutils.resize(img, width=1366)
    cv2.imshow(image_name, newimage)
    cv2.waitKey()
    cv2.destroyAllWindows()


class ArrayPair:
    first_box_center: numpy.array(2)
    second_box_center: numpy.array(2)
    angle: float
    has_angle: bool

    def __init__(self):
        self.has_angle = False

    def calculate_angle(self):
        if self.first_box_center[0] > self.second_box_center[0]:
            self.first_box_center, self.second_box_center = self.second_box_center, self.first_box_center

        self.angle = math.degrees(math.atan2(self.first_box_center[1] - self.second_box_center[1],
                                             self.second_box_center[0] - self.first_box_center[0]))
        self.has_angle = True

    @staticmethod
    def get_pair(first, second, pair_array):
        for i in range(pair_array.__len__()):
            if (pair_array[i][0] == first or pair_array[i][0] == second) and (pair_array[i][1] == first or pair_array[i][1] == second):
                return pair_array[i][2]


def filter_screws(image, screws):
    print(str(screws.shape[0]) + ' screws detected')

    centro_parafusos = []

    # Filtro com houghLines
    for (x, y, w, h) in screws:
        center = (x+w//2, y+h//2)
        centro_parafusos.append(center)

    if centro_parafusos.__len__() == 4:
        return centro_parafusos

    ### Filtro por ângulos ###
    pair_array = numpy.ndarray(shape=(0, 3))

    for i in range(centro_parafusos.__len__() - 1):
        for j in range(i+1, centro_parafusos.__len__()):
            pair = ArrayPair()
            pair.first_box_center = centro_parafusos[i]
            pair.second_box_center = centro_parafusos[j]
            pair.has_angle = False
            ### pair.calculate_angle(angle)

            pair_array = numpy.append(pair_array, [[i, j, pair]], axis=0)

    sup_esq, sup_dir, inf_esq, inf_dir = [], [], [], []
    centroImagem = (image.shape[1] / 2, image.shape[0] / 2)
    i = 0

    ### Coloca as detecções em quadrantes ###
    for (x, y) in centro_parafusos:
        if x < centroImagem[0]:
            if y > centroImagem[1]:
                sup_esq.append((x, y, i))
            else:
                inf_esq.append((x, y, i))
        else:
            if y > centroImagem[1]:
                sup_dir.append((x, y, i))
            else:
                inf_dir.append((x, y, i))
        i += 1

    ### Cria grupos de 4 parafusos, um de cada quadrante ###
    groups = []
    for i in range(sup_esq.__len__()):
        for j in range(sup_dir.__len__()):
            for k in range(inf_esq.__len__()):
                for l in range(inf_dir.__len__()):
                    groups.append((sup_esq[i][2], sup_dir[j][2], inf_esq[k][2], inf_dir[l][2], 0))

    ### Soma os ângulos ###
    newgroup = []
    for iterator in range(groups.__len__()):
        (i, j, k, l, ang_sum) = groups[iterator]

        upper_pair = ArrayPair.get_pair(i, j, pair_array)
        if not upper_pair.has_angle:
            upper_pair.calculate_angle()

        pair = ArrayPair.get_pair(i, k, pair_array)
        if not pair.has_angle:
            pair.calculate_angle()
        ang_sum += math.fabs(upper_pair.angle - pair.angle - 74)

        pair = ArrayPair.get_pair(j, l, pair_array)
        if not pair.has_angle:
            pair.calculate_angle()
        ang_sum += math.fabs(upper_pair.angle - pair.angle - 74)

        pair = ArrayPair.get_pair(k, l, pair_array)
        if not pair.has_angle:
            pair.calculate_angle()
        ang_sum += math.fabs(upper_pair.angle - pair.angle)

        newgroup.append((i, j, k, l, ang_sum))


    ### Organiza os grupos por soma dos ângulos ###
    newgroup = sorted(newgroup, key=operator.itemgetter(4))
    print("Newgroup: "+str(newgroup))

    try:
        (a, b, c, d) = (newgroup[0][0], newgroup[0][1], newgroup[0][2], newgroup[0][3])
    except:
        return None

    filtered_screws = (centro_parafusos[a], centro_parafusos[b], centro_parafusos[c], centro_parafusos[d])
    return filtered_screws