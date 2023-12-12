import torch
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import save_mask

from lightning.pytorch.loggers import TensorBoardLogger
from bc_trav.models import SegmentationModel, fpn_head


class TraversabilityDataset(torch.utils.data.Dataset):
    def __init__(self, src_paths, trg_paths, img_dim = (224,224), transform=None, datasets=['recon']):
        """ 
        Construct an indexed list of image paths and labels
        """
        self.src_paths, self.trg_paths = src_paths, trg_paths 
        self.transform = transform
        self.img_dim = img_dim
        if 'sacson' in datasets:
            self.interp = True # If sacson in datasets, then some images will be (120, 160), so need to upsample

    def __getitem__(self, n):
        """ 
        Load the nth image in the dataset
        Requires: self.trg_paths are .pt files, self.src_paths are .jpg or .png files
        """
        src =  torchvision.io.read_image(self.src_paths[n]).float()/256

        trg = torch.load(self.trg_paths[n])

        if self.interp:
            src = F.interpolate(src[None], size=self.img_dim, mode='bilinear')[0]
            trg = F.interpolate(trg[None], size=self.img_dim, mode='bilinear')[0]

        if self.transform:
            src = self.transform(src)
            trg = self.transform(trg)
            

        return src, trg

    def __len__(self):
        """ return the total number of images in this dataset """
        return len(self.src_paths)

def shuffle_paths(src_paths, trg_paths):
    order = torch.randperm(len(src_paths))
    src_paths = src_paths[order]
    trg_paths = trg_paths[order]
    return src_paths, trg_paths

def get_paths(base_src, base_trg, datasets):
    """
    Structure of base_src/base_trg is as follows:
            base_{src, trg}
            |
            | --- dataset
                |
                |---rollout_{n}
                    |
                    |-- xxxxxx.{jpg, png, pt} 
        
        base_src and base_trg must have the exact same structure,
        src files are in .png or .jpg format, and trg fils are in .pt format
        src may contain additional images that are not being used for training (only files in trg will be loaded)
    """
    src_paths = []
    trg_paths = []

    def get_all_tensor_names(dir):
        return [t for t in os.listdir(dir) if t.lower().endswith(('.pt'))]

    for dataset in os.listdir(base_trg):
        if dataset in datasets:
            for rollout in os.listdir(os.path.join(base_trg, dataset)):

                names = get_all_tensor_names(os.path.join(base_trg, dataset, rollout))
                new_trg = [os.path.join(base_trg, dataset, rollout, n) for n in names]

                new_src = []
                for n in names:
                    new_src_path = os.path.join(base_src, dataset, rollout, n[:-2] + 'png')
                    if not os.path.exists(new_src_path):
                        new_src_path = os.path.join(base_src, dataset, rollout, n[:-2] + 'jpg')
                    new_src.append(new_src_path)

                src_paths += new_src
                trg_paths += new_trg

    #src_paths, trg_paths = shuffle_paths(src_paths, trg_paths)
    return src_paths, trg_paths

def get_train_val_datasets(base_src, base_trg, datasets, transform, split=.85):
    src_paths, trg_paths= get_paths(base_src, base_trg, datasets)
    split_loc = int(len(src_paths) * split)
    train_data = TraversabilityDataset(src_paths[:split_loc], trg_paths[:split_loc], img_dim=(224,224), transform= transform, datasets=datasets )
    val_data = TraversabilityDataset(src_paths[split_loc:], trg_paths[split_loc:], img_dim=(224,224), transform=transform, datasets=datasets)
    return train_data, val_data

def train(model, batch_size, epochs, lr, datasets=['recon'], backbone_id = 'fastervit'):
    
    if 'sacson' in datasets:
        transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
        ])
    else:   
        transform = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop((224, 224))
            ])
    train, val = get_train_val_datasets(base_src = '/home/pcgta/Documents/bc_trav/bc_trav/distill_data/full_data',
                                        base_trg = '/home/pcgta/Documents/bc_trav/bc_trav/distill_data/full_data_preds',
                                        datasets=datasets,
                                        transform = transform,
                                        split=.85)
    

    train_loader = torch.utils.data.DataLoader(train,
                                               batch_size = batch_size,
                                               num_workers = 10,
                                               shuffle = True)


    val_loader = torch.utils.data.DataLoader(val,
                                               batch_size = batch_size,
                                               num_workers = 10,
                                               shuffle = False)



    checkpoint_file=  backbone_id + '-{epoch}' + f'-b={batch_size}-lr={lr:.0e}' + '_'.join(datasets)
    checkpoint_callback = ModelCheckpoint(dirpath='trav_checkpoints/',filename=checkpoint_file, every_n_epochs=1, monitor ='train_loss')



    logger = TensorBoardLogger("tb_logs", name=backbone_id)
    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)



def train_fastervit():

    from model_factories import faster_vit_factory

    backbone_path="fastervit_checkpoints/fastervit_0_224_1k.pth.tar"
    VPT = True
    VPT_PROMPT_LENGTH = 10
    backbone = faster_vit_factory(pretrained=True, vpt=VPT, vpt_prompt_length=VPT_PROMPT_LENGTH,model_path = backbone_path,freeze=True)

    BATCH_SIZE = 16
    EPOCHS = 60
    LR = 6e-3
    IMG_DIM = (224, 224)
    BACKBONE_ID='fastervit'
    DATASETS = ['recon', 'sacson','kitti','asrl']
    
    head = fpn_head(embed_dims = [64, 128, 256, 512], img_height=IMG_DIM[0], img_width=IMG_DIM[1])
    
    
    if hasattr(backbone, 'trainable_params'):
        params =  list(head.parameters()) + backbone.trainable_params
    else:
        params = head.parameters()

    model = torch.nn.Sequential(backbone, head)
    trainable_model = SegmentationModel(model=model, img_dim=IMG_DIM,params = params, lr = LR)
    
    train(trainable_model,BATCH_SIZE,EPOCHS,LR, DATASETS, BACKBONE_ID)

def train_fastscnn():
    from fastscnn.models.fast_scnn import get_fast_scnn

    BATCH_SIZE = 8
    EPOCHS = 60
    LR = 6e-3
    IMG_DIM = (224, 224)
    BACKBONE_ID='scnn'
    DATASETS = ['sacson', 'recon','kitti','asrl']

    model = get_fast_scnn('citys', pretrained=True, root='/home/pcgta/Documents/fastscnn/fastscnn/weights')
    model.activate_n_class_training(2) # change the model classifier to be 2 class instead of 20 class (cityscape default)
    trainable_model = SegmentationModel(model=model, img_dim=IMG_DIM, lr = LR)



    train(trainable_model,BATCH_SIZE,EPOCHS,LR, DATASETS, BACKBONE_ID)

if __name__=="__main__":
    train_fastervit()
    #train_fastscnn()
    pass