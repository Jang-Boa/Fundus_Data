import cv2 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

def create_target_mask(size = 1024, pad = 78):
    """ Target mask to crop fundus image into a perfect circle 
    image size: default 1024 -> is the best to use in this processing"""
    image_size = (size,size,3)
    center = (size//2, size//2)
    radian = 512 + pad
    target_mask = np.zeros(image_size, np.uint8)
    target_mask = cv2.circle(target_mask, center, radian, (1,1,1), -1)
    return target_mask

target_mask = create_target_mask() # Target_mask to use 

def extract_mask(img_pth):
    """ Preprocessing fundus image and generate mask image of fundus """
    image = Image.open(img_pth).convert("RGB")
    image = np.array(image)
    
    ret, imthres = cv2.threshold(image, 12, 1, cv2.THRESH_BINARY)# Extract foreground
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    closing = cv2.morphologyEx(imthres, cv2.MORPH_CLOSE, k)# Extract foreground mask 
    
    new_img = image*closing # remove background
    
    width_min = np.where(closing==1)[1].min()
    width_max = np.where(closing==1)[1].max()

    height_min = np.where(closing==1)[0].min()
    height_max = np.where(closing==1)[0].max()

    new_img = new_img[height_min:height_max, width_min:width_max,:]
    mask = closing[height_min:height_max:, width_min:width_max,:]
    
    h, w = new_img.shape[0], new_img.shape[1]
    
    if h < w: 
        radius = w // 2
        margin = (w - h)
        t = len(np.where(mask[0,:,0]==1)[0])
        b = len(np.where(mask[h-1,:,0]==1)[0])
        if (t==0) | (b==0):
            new_img = new_img
            mask = mask
        else:
            t_m = round(radius - math.sqrt(radius**2 - (t//2)**2))
            b_m = round(radius - math.sqrt(radius**2 - (b//2)**2))
            new_img = cv2.copyMakeBorder(new_img, t_m, b_m, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            mask = cv2.copyMakeBorder(mask, t_m, b_m, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    else:
        radius = h //2
        margin = (h - w)
        l = len(np.where(mask[:,0,0]==1)[0])
        r = len(np.where(mask[:,w-1,0]==1)[0])
        if (l==0) | (r==0):
            new_img = new_img
            mask = mask
        else:
            l_m = round(radius - math.sqrt(radius**2 - (l//2)**2))
            r_m = round(radius - math.sqrt(radius**2 - (r//2)**2))
            new_img = cv2.copyMakeBorder(new_img, 0, 0, l_m, r_m, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            mask = cv2.copyMakeBorder(mask,  0, 0, l_m, r_m, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return new_img, mask[:,:,0]

def crop_with_mask(image, mask, refer_mask=target_mask, pad=88):
    image = cv2.resize(image, (mask.shape[1], mask.shape[0]))
    refer_mask = cv2.copyMakeBorder(refer_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    frac_h = mask.shape[0] / refer_mask.shape[0]
    frac_w = mask.shape[1] / refer_mask.shape[1]
    refer_mask = cv2.resize(refer_mask, (0, 0), fx=frac_w, fy=frac_h)
    
    image = image * refer_mask
    image = image[round(pad * frac_h):-round(pad * frac_h),round(pad * frac_w):-round(pad * frac_w),:]
    mask = mask[round(pad * frac_h):-round(pad * frac_h),round(pad * frac_w):-round(pad * frac_w)]
    
    return image, mask 


