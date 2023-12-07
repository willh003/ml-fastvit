
from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import csv
import os
from torchvision import transforms
from bc_trav.model_factories import ovt_factory
from bc_trav.utils import *


class BC(pl.LightningModule):

    def __init__(self, backbone, feature_dim, action_dim,lr, loss=F.mse_loss):
        super().__init__()

        self.backbone = backbone
        self.head = self.create_head(feature_dim, action_dim)
        self.loss = loss
        self.lr = lr

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

    def validation_step(self, batch, batch_idx):
        src, trg = batch

        probs = self.forward(src)
        loss = self.loss(probs, trg) 
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        optim = torch.optim.AdamW(head_params + backbone_params, lr=self.lr)

        return optim

    def disable_grads(self):
        for param in self.parameters():
            param.requires_grad = False

    def create_head(self, in_dim, out_dim):

        *in_dim, = in_dim
        flattened_dim = np.prod(in_dim)
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.Linear(256, out_dim),    
        )
    
class BCDataset(torch.utils.data.Dataset):
            
    def __init__(self, image_paths, device='cuda'):
        """
        Inputs:
            img_paths: a list of paths to images
                - images are stored as "root/{num},{cmd1},{cmd2},{cmd3}.pt"
            
            transform: the transformation to apply when loading
        """
        self.image_paths = image_paths
        self.device = device

    def __getitem__(self, n):
        """
        Assumes images are stored as "root/{num},{fwd},{lat},{theta}.pt"
        """

        obs = torch.load(self.image_paths[n])
        action = self.image_paths[n].split('/')[-1].split('.')[0].split(",")
        i, fwd, lat, th = action
        
        return obs, float(th)

    def getitem_csv(self, n):
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


def get_train_val_datasets(base, transform, split):
    paths = os.listdir(base)

    split_loc = int(len(paths) * split)
    train_data = BCDataset(paths[:split_loc], paths[:split_loc])
    val_data = BCDataset(paths[split_loc:], paths[split_loc:])
    return train_data, val_data

def train(backbone, cfg):

    batch_size = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    lr = float(cfg['training']['lr'])

    # use huber loss to reduce effect of outliers
    model = BC(backbone, feature_dim=(224, 224), action_dim=1, lr = 3e-4, loss=F.huber_loss)
    breakpoint()


    train, val = get_train_val_datasets(base = '/home/pcgta/Documents/playground/distill/full_data',
                                        split=.85)

    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size = batch_size,
                                               num_workers = 10,
                                               shuffle = True)


    val_loader = torch.utils.data.DataLoader(val,
                                               batch_size = batch_size,
                                               num_workers = 10,
                                               shuffle = False)

    



    checkpoint_file=  'bc' + '-{epoch}-{step}' + f'-b={batch_size}-lr={lr:.0e}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='bc_checkpoints/',filename=checkpoint_file, every_n_epochs=5, monitor ='train_loss')



    logger = TensorBoardLogger("tb_logs", name="bc")
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



if __name__=='__main__':
    ### Demo on how to use ovt_factory to create an ovt model
    trav_cfg_path = 'configs/tuned_fastervit.yaml'
    backbone = ovt_factory(trav_cfg_path)

    train_cfg_path = 'configs/bc_train.yaml'
    training_cfg = open_yaml(train_cfg_path)

    train(backbone, training_cfg)
