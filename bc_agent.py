
from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT 
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import pandas as pd 
import os
from torchvision import transforms

class BC(pl.LightningModule):

    def __init__(self, vision_encoder, feature_dim, action_dim, loss=F.mse_loss):
        self.vision_encoder = vision_encoder
        self.disable_grads()
        self.head = self.create_head(feature_dim, action_dim)
        self.loss = loss

    def forward(self, x):
        e = self.vision_encoder(x)
        pred = self.head(e)
        return pred

    def training_step(self, batch, batch_idx):
        src, trg = batch

        # backbone is frozen, so disable gradients
        with torch.no_grad():
            e = self.vision_encoder(src)

        pred = self.head(e)
        loss = self.loss(src, trg)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def disable_grads(self):
        for param in self.parameters():
            param.requires_grad = False

    def create_head(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256),
            nn.Linear(256, out_dim),    
        )
    

class BCDataloader():
            
    def __init__(self, data_dir, cols=['ts', 'v', 'theta'], model_dim=(224, 224)):
        """
        data_dir must contain a file named 'action.csv', where its first column is a timestamp
        within data_dir there should be a folder named 'images', containing images of the form '{timestamp}.{jpg, png}'
        there must be a 1-1 correspondence between timestamps in the actions and image folders
        """
        
        self.data_dir = data_dir
        self.trgs = pd.read_csv(os.path.join(self.data_dir, 'action.csv'), names=cols, header=None)
        self.trgs.set_index(['ts'])
        self.cols = cols
        self.image_paths = os.listdir(os.path.join(self.data_dir, 'images'))
        self.transform = transforms.CenterCrop(model_dim),
    

    def __getitem__(self, n):
        """ 
        Load the nth image in the dataset
        self.image_paths are .jpg or .png files
        """
        path = os.path.join(self.data_dir, 'images', self.image_paths[n])

        src =  torchvision.io.read_image(path).float()/256
        if self.transform:
            src = self.transform(src)
        
        ts = self.image_paths[n].split('.')[0]
        trg = self.get_data_from_timestamp(ts)
            

        return src, trg
    
    def __len__(self):
        return len(self.image_paths)

    def get_data_from_timestamp(self, ts):
        data = self.trgs.loc[ts]
        return [data[col][0] for col in self.cols[1:]] # get every given action