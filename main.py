import torch
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from bc_trav.utils import save_mask

from lightning.pytorch.loggers import TensorBoardLogger

from bc_trav.models import SegmentationModel
from bc_trav.model_factories import faster_vit_factory, adjust_backbone_bias, trav_prior_factory

def main():
    size=(224,224)
    checkpoint='/home/pcgta/Documents/playground/bc_trav/bc_trav/trav_checkpoints/fastervit-epoch=59-step=125880-b=8-lr=6e-04.ckpt'


    backbone = trav_prior_factory(ckpt=checkpoint, img_dim=size)

    df = "/home/pcgta/Documents/cs6670finalproject/anymalrunner/image_features/test_4"
    for f in os.listdir(df):
        if (f[-1]) == 't':
            continue
        img = torchvision.io.read_image(os.path.join(df, f)).float()/256
        if img.size()[1] < size[0] or img.size()[2] < size[1]:
            img = F.interpolate(img[None], size, mode='bilinear').squeeze(0)

        img = transforms.functional.center_crop(img, size)

        out = backbone(img.cuda()[None]) 
        save_mask(img.cpu().permute(1,2,0), out.argmax(dim=1)[0].float().cpu(), f'/home/pcgta/Documents/playground/bc_trav/bc_trav/image_features_out/{f}_trained.png')