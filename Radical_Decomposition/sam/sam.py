# This file is used to segment the Oracle Bone Inscriptions dataset on GPU 0
import copy
import json
import os
import sys
import time
import argparse
from tool import getmask, getgray, getsamhq, getgenerator0,getgenerator1
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

def maskwrite(masklist,path,num,writepath):
    # Save segmented images based on masks
    image=getgray(path)
    img = Image.fromarray(np.uint8(image)) # Save original gray image
    img = img.convert("RGB")
    img.save(os.path.join(writepath, f'{0}.jpg'))
    image = 255 - image
    for maski in range(0,len(masklist)):
        mask=masklist[maski]
        int_array = mask.astype(np.uint8) * 255
        scaled_int_array = cv2.resize(int_array,(image.shape[1], image.shape[0]))
        mask = (scaled_int_array > 0)
        img = mask * image
        img=255-img
        img = Image.fromarray(np.uint8(img))
        img = img.convert("RGB")
        img.save(os.path.join(writepath,f'{maski+1}.jpg'))

# def showmask(masklist,image):
#     # Display masks one by one
#     image = 255 - image
#     for mask in masklist:
#         img = mask * image
#         img=255-img
#         img = Image.fromarray(np.uint8(img))
#         img = img.convert("RGB")
#         plt.figure()
#         plt.imshow(img)
#         plt.axis('off')
#         plt.show()

# def showmasks(masklist,image):
#     # Display masks from dictionary format
#     image = 255 - image
#     for mask in masklist:
#         mask = mask['segmentation']
#         img = mask * image
#         img=255-img
#         img = Image.fromarray(np.uint8(img))
#         img = img.convert("RGB")
#         plt.figure()
#         plt.imshow(img)
#         plt.axis('off')
#         plt.show()

def inmask(mask1,mask2):
    # Calculate intersection between two masks
    mask=mask1*mask2
    flag=sum(map(sum, mask))
    return flag

def same(mask1,mask2):
    # Check if two masks are nearly the same
    mask=mask1*mask2
    flag=sum(map(sum, mask))
    flag1=sum(map(sum, mask1))
    flag2 = sum(map(sum, mask2))
    if flag>flag1-20 and flag>flag2-20:
        return 1
    return 0

def bigger(mask1,mask2):
    # Return which mask covers a larger area
    mask=mask1*mask2
    mask1=mask1^mask
    mask2 = mask2 ^ mask
    flag1=sum(map(sum, mask1))
    flag2 = sum(map(sum, mask2))
    if(flag2>flag1):
        return 2
    else:
        return 1

def another_part(masklist,masklunko):
    # Get the remaining part of the image after subtraction
    masklistnew=[]
    for mask in masklist:
        masknew=Subtraction(copy.deepcopy(masklunko),mask)
        masklistnew.append(mask)
        masklistnew.append(copy.deepcopy(masknew))
    return masklistnew

def Subtraction(mask1,mask2):
    # Subtract mask2 area from mask1
    mask2 = mask2.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask2 = cv2.dilate(mask2, kernel, iterations=2)
    mask2 = mask2.astype(np.bool_)
    mask = mask1 * mask2
    mask = mask.astype(np.int32)
    mask1 = mask1.astype(np.int32)
    mask1=mask1-mask
    mask1 = mask1.astype(np.bool_)
    return mask1

def process_mask(masklist):
    # Process masks, retain smaller regions
    de=[]
    for i in range(0,len(masklist)-1):
        for j in range(i+1, len(masklist)):
            if j in de or i in de:
                continue
            if same(masklist[i], masklist[j]):
                de.append(j)
                continue
            flag = inmask(masklist[i], masklist[j]) # intersection
            if flag > 50:
                big = bigger(masklist[i], masklist[j])
                if big == 1:
                    mask = masklist[i] * masklist[j]
                    masklist[i]=Subtraction(masklist[i],mask)
                else:
                    mask = masklist[i] * masklist[j]
                    masklist[j]=Subtraction(masklist[j],mask)
    newmasklist=[]
    for i in range(len(masklist)):
        if i not in de:
            newmasklist.append(masklist[i])
    return newmasklist

def process_maskbig(masklist):
    # Process masks, retain larger regions
    de=[]
    for i in range(0,len(masklist)-1):
        for j in range(i+1, len(masklist)):
            if j in de or i in de:
                continue
            if same(masklist[i], masklist[j]):
                de.append(j)
                continue
            flag = inmask(masklist[i], masklist[j])# Take the intersection
            if flag > 50:
                big = bigger(masklist[i], masklist[j])
                if big == 1:
                    de.append(j)
                else:
                    de.append(i)
    newmasklist=[]
    for i in range(len(masklist)):
        if i not in de:
            newmasklist.append(masklist[i])
    return newmasklist

def large(mastlist):
    # Dilate masks to expand regions
    mastlistnew=[]
    for mask in mastlist:
        mask = mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = mask.astype(np.bool_)
        mastlistnew.append(copy.deepcopy(mask))
    return mastlistnew
    
def generatemask(image,masks):
    # Filter masks based on relative area ratio
    flag=0
    masklist=[]
    image = 255 - image
    st= np.sum(image)
    for maski in masks:
        s=0  # total pixels in masked area
        mask = maski['segmentation']
        img = mask * image
        s = np.sum(img)
        s=s/st
        if s > 0.1 and s <0.9:  # valid masks
            masklist.append(copy.deepcopy(maski))
            flag = flag + 1
    return masklist,flag

def get_outline(image):
    # Generate rough outline mask
    min1 = np.min(image)
    max1 = np.max(image)
    image = (image - min1) / (max1 - min1) * 255
    mask = np.where(image < 200, 1, 0)
    return mask

def is_empty(masklist,image):
    # Remove empty or invalid masks
    min1 = np.min(image)
    max1 = np.max(image)
    image = (image - min1) / (max1 - min1) * 255
    image = 255 - image
    de=[]
    for maski in range(0,len(masklist)):
        mask=masklist[maski]
        img = mask * image
        s = 0 
        s = np.sum(img)
        cont = sum(map(sum, mask))
        si=s/cont
        if si<100 or s<15000:
            de.append(maski)
    newmasklist=[]
    for i in range(len(masklist)):
        if i not in de:
            newmasklist.append(masklist[i])
    return newmasklist

def stability(masklist,edge):
    # Keep masks with stability and IoU above threshold
    newmasklist=[]
    for i in masklist:
        if i['predicted_iou']>edge and i['stability_score']>edge:
            newmasklist.append(i)
    return  newmasklist

def mmm(masklist):
    # Extract only the 'segmentation' from mask dictionaries
    newmasklist=[]
    for i in masklist:
        newmasklist.append(i['segmentation'])
    return newmasklist

def main():
    # -------- Parse command line arguments --------
    parser = argparse.ArgumentParser(description="Oracle Bone Segmentation Script")
    parser.add_argument("--json_file", type=str, default="../data/OBI.json",
                        help="Path to input JSON file (default: ../data/OBI.json)")
    parser.add_argument("--output_dir", type=str, default="../output/Decomposition/sam",
                        help="Directory to save segmentation results (default: ../output/Decomposition/sam)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id (default: 0)")
    parser.add_argument("--data_path", type=str, default="../data",
                        help="Root folder of images, will be prepended to JSON paths (default: ../data)")
    args = parser.parse_args()
    # -------- Prepare model and output folders --------
    sam = getsamhq(args.gpu)  # Init SAM on GPU
    with open(args.json_file, 'r', encoding='utf8') as f:
        jgw = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    # -------- Process each image --------
    for num in tqdm(range(len(jgw))):
        path = os.path.join(args.data_path, jgw[num]['path'])  # prepend data_path
        label = jgw[num]['label']
        img = getgray(path)
        filename_with_extension = os.path.basename(path)
        filename_without_extension = os.path.splitext(filename_with_extension)[0]

        # Create output directories
        save_dir = os.path.join(args.output_dir, label, filename_without_extension)
        os.makedirs(save_dir, exist_ok=True)

        # Use different generators depending on filename
        if filename_with_extension[0] not in ['H', 'Y']:
            masks = getmask(img.copy(), getgenerator1(sam))
        else:
            masks = getmask(img.copy(), getgenerator0(sam))

        masklist, flag = generatemask(img.copy(), masks)
        masklist = mmm(masklist)
        masklunko = get_outline(img.copy())
        masklist = process_maskbig(masklist)
        masklist = another_part(masklist, masklunko)
        masklist = process_mask(masklist)
        masklist = is_empty(masklist, img.copy())
        if len(masklist) == 1:
            masklist = another_part(masklist, masklunko)
            masklist = is_empty(masklist, img.copy())
        masklist = large(masklist)
        maskwrite(masklist, path, num, save_dir)

if __name__ == "__main__":
    main()
