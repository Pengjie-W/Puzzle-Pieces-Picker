import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image, ImageFont, ImageDraw
def getgray(path):

    # Use PIL to open the PNG image and convert it to grayscale
    image = Image.open(path)

    if image.mode == 'RGBA':
        white_bg = Image.new('RGB', image.size, (255, 255, 255))
        white_bg.paste(image, mask=image.split()[3])  
        gray_image = white_bg.convert('L')

    else:
        gray_image = image.convert('L')
    img = np.array(gray_image)

    return img

def getmask(img,mask_generator):
    img=Image.fromarray(np.uint8(img))
    img=img.convert("RGB")
    image = np.array(img)
    masks = mask_generator.generate(image)
    return masks
def getgenerator1(sam):
    mask_generator= SamAutomaticMaskGenerator(
        model=sam,        points_per_side=32,
        points_per_batch=128,
        box_nms_thresh=0.98,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.95,
    )
    return mask_generator
def getgenerator0(sam):
    mask_generator= SamAutomaticMaskGenerator(
        model=sam,        points_per_side=32,
        points_per_batch=128,
        box_nms_thresh=0.95,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.95,
    )
    return mask_generator

def getsam(num):
    model_type = "vit_h"
    device = f"cuda:{num}"
    sam_checkpoint = "./sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return sam
    
def getsamhq(num):
    from segment_anything import sam_model_registry
    model_type = "vit_h"
    device = f"cuda:{num}"
    sam_checkpoint = "./sam_hq_vit_h.pth"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint, map_location=torch.device(device))
    sam.to(device=device)
    return sam