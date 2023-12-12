
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
from bc_trav.model_factories import get_factory_from_type
from bc_trav.utils import *

from pathlib import Path


    
class BCDataset(torch.utils.data.Dataset):
            
    def __init__(self, image_paths, device='cuda'):
        """
        Inputs:
            img_paths: a list of paths to images
                - images are stored as "root/{rollout},{num},{cmd1},{cmd2},{cmd3}.pt"
            
            transform: the transformation to apply when loading
        """
        self.image_paths = self.sort_by_nth_id(image_paths, rollout_n=0, obs_n=1)

        self.device = device
    
        # get all sequences of 6
    
        self.all_sixes = []
        theta_counts = torch.zeros(3)
        for i in range(len(image_paths)-6): #might be -5, but this won't break
            new_six = []
            first_in_batch = image_paths[i]
            last_in_batch = image_paths[i+5]
            rl1, _,_,_,th = first_in_batch.split("/")[-1].split('.pt')[0].split(",")
            rl6, _,_,_,_ = last_in_batch.split("/")[-1].split('.pt')[0].split(",")
            if rl1 == rl6:
                [new_six.append(image_paths[i+j]) for j in range(6)]
                self.all_sixes.append(new_six)
            
            theta_counts += self.get_theta_one_hot(float(th))

        self.class_weight = torch.div(torch.max(theta_counts),theta_counts)
        print(f'Class weight: {self.class_weight}')

                
    def get_theta_one_hot(self, th):

        if th < 0:
            th_one_hot = [1, 0, 0]
        elif th == 0:
            th_one_hot = [0,1,0]
        else:
            th_one_hot = [0,0,1]
        th_one_hot = torch.as_tensor(th_one_hot).float()
        return th_one_hot

    def __getitem__(self, n):
        """
        Assumes images are stored as "root/{num},{fwd},{lat},{theta}.pt"
        """
        image_paths = self.all_sixes[n]

        # most recent theta is on top (image_paths[-1])
        # theta stored as 5th value of filename
        th = image_paths[-1].split('/')[-1][:-3].split(",")[4].split(".pt")[0]
        th_one_hot = self.get_theta_one_hot(float(th))
        
        obs = torch.stack([torch.load(image_paths[i]) for i in range(6)])
        obs = obs.permute(0, 3,1,2)
        
        return obs.float(), th_one_hot
    
    def sort_by_nth_id(self, paths, rollout_n, obs_n):
        list.sort(paths, key = lambda x: 10000*int(x.split('/')[-1][:-3].split(",")[rollout_n])+ int(x.split('/')[-1][:-3].split(",")[obs_n]))
        return paths

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
        return len(self.all_sixes)

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


def get_train_val_datasets(base, split):
    paths = [os.path.join(base, f) for f in os.listdir(base)]
    
    data = BCDataset(paths)
    class_weights = data.class_weight
    split_loc = int(len(data.all_sixes) * split)
    train_paths = data.all_sixes[:split_loc]
    val_paths = data.all_sixes[split_loc:]

    train_data = BCDataset([point for rollout in train_paths for point in rollout ])
    val_data = BCDataset([point for rollout in val_paths for point in rollout ])
    return train_data, val_data, class_weights

def train(trav_cfg_path, train_cfg_path):

    cfg= open_yaml(train_cfg_path)
    batch_size = cfg['training']['batch_size']
    epochs = cfg['training']['epochs']
    lr = cfg['training']['lr']
    base_df = cfg['training']['df']
    device = cfg['training']['device']
    #model_type = cfg['model_type']

    ### ONLY FOR OVERNIGHT TRAINING:

    for model_type in ['trav', 'image', 'fusion']:
        
        train, val, class_weights = get_train_val_datasets(base = base_df,
                                            split=.85)

        train_loader = torch.utils.data.DataLoader(train,
                                                batch_size = batch_size,
                                                num_workers = 10,
                                                shuffle = True,
                                                drop_last=True)
        val_loader = torch.utils.data.DataLoader(val,
                                                batch_size = batch_size,
                                                num_workers = 10,
                                                shuffle = False,
                                                drop_last=True)

        checkpoint_file=model_type + '-{epoch}-{step}' + f'-b={batch_size}-lr={lr}'

        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='bc_checkpoints/',filename=checkpoint_file, every_n_epochs=1, monitor ='train_loss')

        logger = TensorBoardLogger("tb_logs", name="bc")
        trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], logger=logger)


        factory = get_factory_from_type(model_type)
        
        model = factory(trav_cfg_path, train_cfg_path,class_weights=class_weights, device=device, lr=float(lr))


        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



if __name__=='__main__':
    ### Demo on how to use ovt_factory to create an ovt model
    trav_cfg_path = 'configs/tuned_fastervit.yaml'
    train_cfg_path = 'configs/bc_train.yaml'
    
    train(trav_cfg_path, train_cfg_path)

