# This code uses OpenCV to segment Oracle Bone script radicals
import copy
import json
import os
import math
import random
from toolscomplex import sort_contours
import cv2 
import numpy as np
from toolscomplex import mincontour
from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Oracle Bone script segmentation using OpenCV")
parser.add_argument("--mode", type=str, default="run", help="Mode: run or test")
parser.add_argument("--data_path", type=str, default="../data", help="Path to input OBI dataset")
parser.add_argument("--json_file", type=str, default="../data/OBI.json", help="Path to OBI json file (used in run mode)")
parser.add_argument("--output_dir", type=str, default="../output/Decomposition/opencv", help="Output directory for segmentation results")
args = parser.parse_args()

MODE = args.mode

link_max_size_rate = 5/(400*400)
link_min_distance_rate = 20/(400*400)
link_max_gray_value = 200

MAX_POINT_SIZE_RATE = 0.006
MAX_IGNORE_SIZE_RATE = 0.0007

def getgray(path):
    image = Image.open(path)
    if image.mode == 'RGBA':
        white_bg = Image.new('RGB', image.size, (255, 255, 255))
        white_bg.paste(image, mask=image.split()[3])
        gray_image = white_bg.convert('L')
    else:
        gray_image = image.convert('L')
    img = np.array(gray_image)
    return img

def cv_show(img, name):# Display image
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def hull_cut(filter_xy,img,dianlist,dianarea,dianji):  # Crop image by convex hull
    hull=cv2.convexHull(filter_xy)
    col0 =hull[:,:,0]
    col1 =hull[:,:,1]
    x1=np.min(col0)
    y1=np.min(col1)
    x2=np.max(col0)
    y2 = np.max(col1)
    if x1 < 0:
       x1 = 0
    if x2 > img.shape[1]:
       x2 = img.shape[1]
    if y1 < 0:
        y1 = 0
    if y2 > img.shape[0]:
        y2 = img.shape[0]
    if (cv2.contourArea(filter_xy) == dianarea) and (dianarea != 0) and np.array_equal(filter_xy, dianji): # If this is a set of points
        mask = np.zeros(img.shape, dtype=np.uint8)  # Create a mask of the same size as the original image, all initialized to 0
        for i in range(0, len(dianlist)):
            if (i == 0):
                be = 0
                en = dianlist[0]
                mask2 = cv2.fillPoly(mask, [filter_xy[be:en]], (255, 255, 255))
            else:
                be = en
                en = dianlist[i]+en
                mask2 = cv2.fillPoly(mask2, [filter_xy[be:en]], (255, 255, 255))
    else:
        mask = np.zeros(img.shape,dtype=np.uint8)  # Create a mask of the same size as the original image, all initialized to 0
        mask2 = cv2.fillPoly(mask,[filter_xy],(255,255,255))
        kernel = np.ones((3, 3), np.uint8)
        mask2 = cv2.dilate(mask2, kernel, iterations=1)
        # mask2 = cv2.fillPoly(mask,[filter_xy],(255,255,255))
    ROI = cv2.bitwise_and(mask2, img)
    # img_end=ROI[y1:y2,x1:x2]
    img_end=ROI
    return img_end

def distance(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

def cut_img(cnt, gray, point_num):
    link_max_size = link_max_size_rate*gray.shape[0]*gray.shape[1]
    link_min_distance = math.floor(link_min_distance_rate*gray.shape[0]*gray.shape[1])
    max_point_size = MAX_POINT_SIZE_RATE*gray.shape[0]*gray.shape[1]

    points = []
    ditemp = link_max_size
    djtemp = link_max_size
    flag = 0
    for i in range(cnt.shape[0]):
        if flag:
            flag=flag-1
            continue
        for j in range(i+link_min_distance, min(cnt.shape[0]-link_min_distance+i, cnt.shape[0])):
            t = distance(cnt[i][0], cnt[j][0])
            if (t < link_max_size):
                if (t < djtemp):
                    djtemp = t
                    pjtemp = [cnt[i][0], cnt[j][0]]
                else:
                    if (djtemp < ditemp):
                        ditemp = djtemp
                        pitemp = pjtemp.copy()
                    else:
                        points.append(pitemp.copy())
                        ditemp = link_max_size
                        flag = link_min_distance
                    djtemp = link_max_size
                    break

    flag = 0
    for i in points:
        centerx = (i[0][0] + i[1][0]) // 2
        centery = (i[0][1] + i[1][1]) // 2
        if gray[centery][centerx] < link_max_gray_value:
            gray_temp = gray.copy()
            cv2.line(gray_temp, i[0], i[1], 255, 2)
            
            if MODE == 'test':
                cv2.imencode('.jpg', gray)[1].tofile('test.jpg')  
            
            contours = get_cnts(gray_temp)
            point_num_temp = 0
            for i in range(0, len(contours)):
                if (cv2.contourArea(contours[i]) < max_point_size):
                    point_num_temp = point_num_temp+1
            if point_num_temp == point_num:
                gray = gray_temp.copy()
                flag = 1

    return gray

def get_cnts(gray):
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    # b: binary image, c: contour info, h: hierarchy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def get_paths(datapath):
    dirs = os.listdir(datapath)
    data = []
    for dir in dirs:
        dirpath = os.path.join(datapath, dir)
        for pic in os.listdir(dirpath):
            pic_path = os.path.join(dirpath, pic)
            if os.path.isdir(pic_path):
                for i in os.listdir(pic_path):
                    data.append(os.path.join(pic_path, i))
            else:
                data.append(pic_path)
    return data

def main():
    data_set = []
    data = {}
    num1 = num2 = num3 = num4 = 0
    if MODE == 'run':
        with open(args.json_file, 'r', encoding='utf8') as f:
            pic_data = json.loads(f.read())
    elif MODE == 'test':
        _labels = os.listdir(os.path.join('OBI',args.data_path))
        random.shuffle(_labels)
        labels = _labels[:50]

    for num in tqdm(range(len(pic_data))):
        path = pic_data[num]['path']
        label = path.split('/')[-2]
        fwe = path.split('/')[-1]

        path=os.path.join(args.data_path,path)
        fwoe = os.path.splitext(fwe)[0] # filename_without_extension
        _writepath = os.path.join(args.output_dir, label)
        writepath = os.path.join(os.path.curdir, _writepath, f'{fwoe}')

        os.makedirs(writepath, exist_ok=True)
        gray = getgray(path)
        cv2.imencode('.jpg', gray)[1].tofile(writepath + '/0.jpg')

        contours = get_cnts(gray)
        if(len(contours)==0):# Skip if no contours are detected
            data['index'] = num
            data['structure'] = 'empty'
            data_set.append(copy.deepcopy(data))
            data.clear()
            continue
        
        max_point_size =  MAX_POINT_SIZE_RATE*gray.shape[0]*gray.shape[1]
        point_num = 0
        for i in range(0, len(contours)):
            if (cv2.contourArea(contours[i]) < max_point_size):
                point_num = point_num + 1
        if fwe[0] not in 'GL':

            for cnt in contours:
                gray = cut_img(cnt, gray.copy(), point_num)
        
        contours = get_cnts(gray)

        draw_img = gray.copy()
        err = 0
        l = []
        for i in range(0, len(contours)):
            l.append(cv2.contourArea(contours[i]))
        zipcounter = zip(l, contours)
        sorted_zip = sorted(zipcounter, key=lambda x: x[0])
        l,contours = zip(*sorted_zip)# Sorted by contour area
        for i in range(0, len(contours)):
            if (cv2.contourArea(contours[i]) < MAX_IGNORE_SIZE_RATE*gray.shape[0]*gray.shape[1]):
                err=err+1
        contours = (contours[err:])# Ignore very small areas

        dianlist=[]
        dian = 0
        dianji=None
        for i in range(0, len(contours)):
            if (cv2.contourArea(contours[i]) <= max_point_size):# Treat very small contours as point sets
                if (dian == 0):
                    dianji = contours[i]
                    dian = dian + 1
                else:
                    dianji = np.vstack((dianji, contours[i]))
                    dian = dian + 1
                dianlist.append(contours[i].shape[0])
            else:
                break

        dianarea=0
        if dian==1: # Special handling if only one point detected
            if len(contours)==2:
                mintemp = np.vstack((contours[0], contours[1]))# Do not cut in this case
                contours=(mintemp,)
            else:
                minindex=mincontour(contours,0)# Choose the nearest contour
                mintemp=contours[minindex]
                mintemp=np.vstack((contours[0],mintemp))
                if minindex == len(contours)-1:
                    contours = (mintemp,) + contours[1:minindex]
                elif(minindex!=1):
                    contours = (mintemp,) + contours[1:minindex]+contours[minindex+1:]
                else:
                    contours = (mintemp,)  + contours[minindex + 1:]
        if dian > 1:
            dianarea = cv2.contourArea(dianji)
            contours = (dianji,) + contours[dian:]# Combine point set with the rest
        l = []# Up to here, removed errors and merged point sets
        for i in range(0, len(contours)):
            chu=0
            chuji = contours[0]
            for j in range(0, len(contours)):
                if (j!=i):
                    if (chu == 0):
                        chuji = contours[j]
                        chu = chu + 1
                    else:
                        chuji = np.vstack((chuji, contours[j]))
                        chu = chu + 1
                else:
                    continue
            (x, y, w, h) = cv2.boundingRect(chuji)
            l.append(w*h)# Compute bounding rectangle excluding this part
        zipcounter = zip(l, contours)
        sorted_zip = sorted(zipcounter, key=lambda x: x[0])
        l,contours = zip(*sorted_zip)# Sort ascending; smaller area means closer to edge
        if(len(contours)==1):
            num1=num1+1
        elif len(contours)==2:
            num2=num2+1
        elif len(contours)==3:
            num3=num3+1
        else:
            num4=num4+1

        for co in range(0,len(contours)):# Save segmented images
            cnt=contours[co]
            gray1=getgray(path)
            gray1 = 255 - gray1
            try:
                new = hull_cut(cnt, gray1,dianlist,dianarea,dianji)
            except:
                print(path)
                continue
            new=255-new
            cv2.imencode('.jpg', new)[1].tofile(writepath + f'/{co+1}.jpg')
        if ((len(contours) == 1)):
            data['index'] = num
            data['structure'] = 'single'
        else:# Extract structure info (currently not used)
            for feni in range(0, len(contours)-1):
                fen = 0
                for i in range(feni+1, len(contours)):
                    if (fen == 0):
                        fenji = contours[i]
                        fen = fen + 1
                    else:
                        fenji = np.vstack((fenji, contours[i]))
                        fen = fen + 1
                if fen != 0:
                    fenji = (contours[feni],) + (fenji,)
                t, structtemp = sort_contours(fenji, feni+1)
                if (feni == 0):
                    structure = structtemp
                else:

                    structure = structure.replace(f'{feni+1}', structtemp)
            data['index'] = num
            data['structure'] =structure
        data_set.append(copy.deepcopy(data))
        data.clear()
    
    data['simply'] = f"{num1}"
    data['double'] = f"{num2}"
    data['three'] = f"{num3}"
    data['complex'] = f"{num4}"
    data_set.append(copy.deepcopy(data))

    if MODE == 'run':
        with open('../output/datacomplexgainew.json', 'w', encoding='utf8') as f:
            json.dump(data_set,
                    f, 
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False)
    elif MODE == 'test':
        with open('../output/test_result.json', 'w', encoding='utf8') as f:
            json.dump(data_set,
                    f,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False)


if __name__ == "__main__":
    main()
