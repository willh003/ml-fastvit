
from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT 
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import csv
import os
from torchvision import transforms
from model_factories import ovt_factory


class BC(pl.LightningModule):

    def __init__(self, backbone, feature_dim, action_dim, loss=F.mse_loss):
        self.backbone = backbone
        self.disable_grads()
        self.head = self.create_head(feature_dim, action_dim)
        self.loss = loss

    def forward(self, x):
        e = self.backbone(x)
        pred = self.head(e)
        return pred

    def training_step(self, batch, batch_idx):
        src, trg = batch

        # backbone is frozen, so disable gradients
        with torch.no_grad():
            e = self.backbone(src)

        pred = self.head(e)
        loss = self.loss(pred, trg)
        
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
            
    def __init__(self, data_dir, cols=['ts', 'v', 'theta'], model_dim=(224, 224), device='cuda'):
        """
        data_dir must contain a file named 'action.csv', where its first column is a timestamp
        within data_dir there should be a folder named 'images', containing images of the form '{timestamp}.{jpg, png}'
        there must be a 1-1 correspondence between timestamps in the actions and image folders
        """
        
        self.data_dir = data_dir
        self.trgs = self.index_action_by_ts(data_dir)
        self.cols = cols
        self.image_paths = os.listdir(os.path.join(self.data_dir, 'images'))
        self.transform = transforms.CenterCrop(model_dim)
        self.device = device

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
        trg = self.trgs[ts]
            
        return src, trg
    
    def __len__(self):
        return len(self.image_paths)

    def index_action_by_ts(self, df):
        index = {}

        with open(os.path.join(df, 'action.csv')) as f:
            reader = csv.DictReader(f, fieldnames=self.cols, delimiter=',', quotechar='|')
            for row in reader:
                ts = row['ts'] # index by ts col
                action = [row[a] for a in self.cols if a!='ts'] # all other cols are actions
                action_tensor = torch.as_tensor(action, dtype=torch.float32)
                index[ts] = action_tensor

        return index

if __name__=='__main__':
    ### Demo on how to use ovt_factory to create an ovt model
    cfg = 'configs/tuned_fastervit.yaml'
    m =  ovt_factory(cfg)
    x = torch.rand((1, 18, 224, 224)) # x must have 18 channels (3 for each image, 6 total images)
    y= m(x)