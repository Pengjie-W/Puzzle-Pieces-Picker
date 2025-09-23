import copy
import os
import sys
import json
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
# Cut out blank background
dataset={}
def getgray(path):
    image = Image.open(path)
    if image.mode == 'RGBA' :
        white_bg = Image.new('RGB', image.size, (255, 255, 255))
        white_bg.paste(image, mask=image.split()[3])
        img=np.array(white_bg)
        gray_image = white_bg.convert('L')
    else:
        image = image.convert('RGB')
        img=np.array(image)
        gray_image = image.convert('L')
    gray = np.array(gray_image)
    return img,gray
with open('./output/Decomposition_Dataset.json', 'r', encoding='utf8') as f:
    data = json.load(f)
for i in tqdm(data):
    path=i['path']
    image = Image.open(path)
    img = np.array(image)
    gray_image = image.convert('L')
    gray = np.array(gray_image)
    err = 0
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    l = []
    for i in range(0, len(contours)):
        l.append(cv2.contourArea(contours[i]))
    zipcounter = zip(l, contours)
    sorted_zip = sorted(zipcounter, key=lambda x: x[0])
    if len(sorted_zip) == 0:
        print(path)
        continue
    l, contours = zip(*sorted_zip)
    if len(contours) > 1:
        for i in range(0, len(contours)):
            if i == 0:
                jihe = contours[0]
            else:
                jihe = np.vstack((jihe, contours[i]))
        (x, y, w, h) = cv2.boundingRect(jihe)
    elif len(contours) == 1:
        (x, y, w, h) = cv2.boundingRect(contours[0])
    if x + w < gray.shape[1]:
        w = w + 1
    if x + w < gray.shape[1]:
        w = w + 1
    if x > 1:
        x = x - 1
        w = w + 1
    if x > 1:
        x = x - 1
        w = w + 1
    if y + h < gray.shape[0]:
        h = h + 1
    if y + h < gray.shape[0]:
        h = h + 1
    if y > 1:
        y = y - 1
        h = h + 1
    if y > 1:
        y = y - 1
        h = h + 1
    cut = img[y:y + h, x:x + w]
    opencv_image_rgb = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB) 
    pil_image_converted = Image.fromarray(opencv_image_rgb)
    pil_image_converted.save(path)