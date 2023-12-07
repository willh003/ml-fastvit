import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 
import yaml 

def open_yaml(path):
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

############ Data Vis #################

def save_binary_mask(img, pred, path="live_updates/live.png"):
    """
    Writes an image with a binary mask to the file given by f
    Inputs:
        img: a torch tensor representing an rgb image
        pred: a torch tensor representing binary predictions
        f: a file path
    """
    img_to_save = img.permute(1,2,0)
    img_to_save = torch.flip(img_to_save, dims=[2])
    img_to_save = img_to_save.cpu().numpy() * 255
    mask_rgb = np.ones_like(img_to_save)
    mask_rgb[:, :, 2] = pred.numpy()*255 


    masked = cv2.addWeighted(img_to_save.astype(np.float32), .6, mask_rgb.astype(np.float32), .4, 0)
    cv2.imwrite(path,masked)

def save_mask(image, pred, path):
    """
    Writes a masked image with any number of classes
    Note that this will not work in Isaac Sim (matplotlib cannot obtain graphics kernel when isim running)
    """

    plt.imshow(image) # I would add interpolation='none'
    plt.imshow(pred, cmap='jet', alpha=0.5) # interpolation='none'
    plt.savefig(path)
