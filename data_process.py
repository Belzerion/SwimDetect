import json
import os
from pathlib import Path
from re import I
import numpy as np
import re
import math
from PIL import Image
import matplotlib.pyplot as plt

from skimage.draw import disk

import config as cfg

class Entree:
    def __init__(self, nageur_1, nageur_2, fichier_1, fichier_2) -> None:
        self.nageur_1 = nageur_1
        self.nageur_2 = nageur_2
        self.fichier_1 = fichier_1
        self.fichier_2 = fichier_2
    def load_patch(self, nageur, fichier):
        full_image = Image.open(fichier.replace('json', 'png'))
        arr = np.array(full_image)
        rectangle = get_rectangle_from_points(nageur)

        rectangle[0] -= 10
        rectangle[1] -= 10
        rectangle[2] += 10
        rectangle[3] += 10

        patch = arr[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]

        img = Image.fromarray(patch)
        img = np.array(img.resize((cfg.width, cfg.height)))

        img = img / 255.0

        if nageur[0][0] < nageur[1][0]:
            return np.fliplr(img)
        
        return img

    def x(self):
        arr = np.zeros((2, cfg.height, cfg.width, 3))
        arr[0] = self.load_patch(self.nageur_1, self.fichier_1)
        arr[1] = self.load_patch(self.nageur_2, self.fichier_2)
        return arr
    
    def y(self):
        arr = np.zeros((cfg.target_height, cfg.target_width, 4))
        nageur = self.nageur_2
        #si la tete est a gauche, flip toutes les coordonnees x
        if self.nageur_2[0][0] < self.nageur_2[1][0]:
            for i, point in enumerate(nageur):
                nageur[i][0] = cfg.full_image_width - point[0]
        rectangle = get_rectangle_from_points(self.nageur_2)
        for i, point in enumerate(nageur[:4]):#certaines etiquettes ont 5 points...
            x = int((point[0] - rectangle[0] + 10) / (rectangle[2] - rectangle[0] + 20) * cfg.target_width) - 1
            y = int((point[1] - rectangle[1] + 10) / (rectangle[3] - rectangle[1] + 20) * cfg.target_height) - 1
            
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            x = cfg.target_width - 1 if x >= cfg.target_width else x
            y = cfg.target_height - 1 if y >= cfg.target_height else y
            arr[y, x, i] = 1
        return arr
    def __str__(self) -> str:
        return self.fichier_1 + ' ' + self.fichier_2

# Pris sur internet :
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(str(value))
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_data():
    DATA_PATH = Path('dataset_label/')

    dirs = os.listdir(DATA_PATH)

    labels_files = {}

    for race in dirs:
        fichiers = sorted(DATA_PATH.glob(race + '/*.json'), key=numericalSort)
        labels_files[race] = []
        for fichier in fichiers:
            labels_files[race].append(str(fichier))

    courses = {}

    for key in labels_files:
        course = {}
        courses[key] = {}
        for fichier in labels_files[key]:
            nageurs = []
            with open(fichier, 'r') as f:
                data = json.load(f)
            for shape in data['shapes']:
                nageurs.append(shape['points'])
            courses[key][fichier] = nageurs

    entrees = []
    for course in courses:
        images_1 = [image for image in [key for key in courses[course]][:-1]]
        images_2 = [image for image in [key for key in courses[course]][1:]]
        # le choix d'association des paires de nageurs est un sujet de débat (rapport et oral)
        for image_1, image_2 in zip(images_1, images_2):
            points_1 = courses[course][image_1]
            points_2 = courses[course][image_2]
            for nageur_1 in points_1:
                distances = []
                for nageur_2 in points_2:
                    distances.append(euclidean_distance(nageur_1[0], nageur_2[0]))

                min_distance = min(distances)
                i = distances.index(min_distance)
                entrees.append(Entree(nageur_1, points_2[i], image_1, image_2))
    
    return entrees
                

def euclidean_distance(points, points_bis):
    dist = math.pow(points[0] - points_bis[0],2) +  math.pow(points[1] - points_bis[1],2)
    return dist

def get_rectangle_from_points(points):
    #x1, y1, x2, y2
    rectangle = [points[0][0], points[0][1], points[0][0], points[0][1]]
    for point in points[1:]:
        if point[0] < rectangle[0]:
            rectangle[0] = point[0]
        if point[0] > rectangle[2]:
            rectangle[2] = point[0]
        if point[1] < rectangle[1]:
            rectangle[1] = point[1]
        if point[1] > rectangle[3]:
            rectangle[3] = point[1]
    rectangle[0] = int(rectangle[0])
    rectangle[1] = int(rectangle[1])
    rectangle[2] = int(rectangle[2])
    rectangle[3] = int(rectangle[3])
    return rectangle



def check_max_size_box(entrees):
    largeurs = []
    hauteurs = []
    for entree in entrees:
        rectangle = get_rectangle_from_points(entree.nageur_1)
        largeurs.append(rectangle[2] - rectangle[0])
        hauteurs.append(rectangle[3] - rectangle[1])
    largeurs = sorted(largeurs)
    hauteurs = sorted(hauteurs)
    
    print(len(largeurs))
    print('medianne largeur: ' + str(largeurs[len(largeurs)//2]))
    print('medianne hauteur: ' + str(hauteurs[len(hauteurs)//2]))
    print('moyenne largeur: ' + str(sum(largeurs) / len(largeurs)))
    print('moyenne hauteur: ' +  str(sum(hauteurs) / len(hauteurs)))
    print(f'min : {min(largeurs)}')
    print(f'min : {min(hauteurs)}')
    print(f'max : {max(largeurs)}')
    print(f'max : {max(hauteurs)}')
    plt.plot(largeurs, c  = 'b')
    plt.plot(hauteurs, c = 'g')
    plt.title('dispersion des hauteurs (courbe verte), largeurs (courbe bleue)')
    plt.show()
        

if __name__ == '__main__':
    inputs = load_data()
    #check_max_size_box(inputs)

    for i in range(10):
        image  = inputs[i].load_patch(inputs[i].nageur_2, inputs[i].fichier_2)
        Image.fromarray(image).show()
        #for j in range(4):
        image = inputs[i].y()[:,:,0]
        plt.imshow(image)
        plt.show()
    
    

