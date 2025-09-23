import cv2
import math


def mincontour(counters,index): # Find the closest one
    if len(counters)<=1:
        return 0
    else:
        lind=[]
        ljuli=[]
        (x2, y2, w2, h2) = cv2.boundingRect(counters[index])
        x2 = x2 + w2 / 2
        y2 = y2 + h2 / 2
        for i in range(0,len(counters)):
            if i!=index:
                lind.append(i)
                (x1, y1, w1, h1) = cv2.boundingRect(counters[i])
                x1 = x1 + w1 / 2
                y1 = y1 + h1 / 2
                juli=math.sqrt(pow((x1-x2),2)+pow((y1-y2),2))
                ljuli.append(juli)
        zipcounter = zip(ljuli, lind)
        sorted_zip = sorted(zipcounter, key=lambda x: x[0])
        ljuli, lind = zip(*sorted_zip)
        return lind[0]
    

def sort_contours(cnts,num):

    reverse = False
    i = 0
    (x1, y1, w1, h1) = cv2.boundingRect(cnts[0])
    (x2, y2, w2, h2) = cv2.boundingRect(cnts[1])
    if(x1<x2+3)and(y1<y2+3)and(x1+w1+3>x2+w2)and(y1+h1+3>y2+h2):
        return 0,f"⿴{num} {num+1}"
    # if (x2 < x1+2) and (y2 < y1+2) and (x2 + w2+2 > x1 + w1) and (y2 + h2+2 > y1 + h1):
    #     return 1, f"⿴{num+1} {num}"
    x1=x1+w1/2
    x2 = x2 + w2 / 2
    y1 = y1 + h1 / 2
    y2 = y2 + h2 / 2
    if(abs(x1-x2)>abs(y1-y2))and(w1/h1<7)and(w2/h2<7):
        if(x1>x2):
            return 1,f"⿰{num+1} {num}"
        else:
            return 0,f"⿰{num} {num+1}"
    elif (h1/w1<7)and(h2/w2<7)and(abs(x1-x2)<=abs(y1-y2)):
        if (y1 > y2):
            return 1,f"⿱{num+1} {num}"
        else:
            return 0,f"⿱{num} {num+1}"
    elif(w1/h1>=7)or(w2/h2>=7):
        if (y1 > y2):
            return 1,f"⿱{num+1} {num}"
        else:
            return 0,f"⿱{num} {num+1}"
    else:
        if(x1>x2):
            return 1,f"⿰{num+1} {num}"
        else:
            return 0,f"⿰{num} {num+1}"
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized