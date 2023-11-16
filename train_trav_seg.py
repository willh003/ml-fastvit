import torch
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from utils import save_mask

from lightning.pytorch.loggers import TensorBoardLogger

from models import SegmentationModel
from model_factories import faster_vit_factory, adjust_backbone_bias


class TraversabilityDataset(torch.utils.data.Dataset):
    def __init__(self, src_paths, trg_paths, transform=None):
        """ 
        Construct an indexed list of image paths and labels
        """
        self.src_paths, self.trg_paths = src_paths, trg_paths 
        self.transform = transform

    def __getitem__(self, n):
        """ 
        Load the nth image in the dataset
        Requires: self.trg_paths are .pt files, self.src_paths are .jpg or .png files
        """
        src =  torchvision.io.read_image(self.src_paths[n]).float()/256

        trg = torch.load(self.trg_paths[n])
        
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
    train_data = TraversabilityDataset(src_paths[:split_loc], trg_paths[:split_loc], transform)
    val_data = TraversabilityDataset(src_paths[split_loc:], trg_paths[split_loc:], transform)
    return train_data, val_data

def train(backbone, batch_size, epochs, lr, img_dim, num_classes, datasets=['recon'], backbone_id = 'fastervit'):
    transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(img_dim),
    ]
)
    train, val = get_train_val_datasets(base_src = '/home/pcgta/Documents/playground/distill/full_data',
                                        base_trg = '/home/pcgta/Documents/playground/distill/full_data_preds',
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


    model = SegmentationModel(backbone=backbone, num_classes = num_classes, img_dim=img_dim,backbone_id=backbone_id, lr = lr)
    checkpoint_file=  backbone_id + '-{epoch}-{step}' + f'-b={batch_size}-lr={lr:.0e}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='trav_checkpoints/',filename=checkpoint_file, every_n_epochs=5, monitor ='train_loss')



    logger = TensorBoardLogger("tb_logs", name=backbone_id)
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def test_images(backbone, checkpoint, size, df, backbone_id='fastervit'):
    h, w = size

    sd = torch.load(checkpoint)['state_dict']
    model = SegmentationModel(num_classes=2, img_dim=(h, w), backbone=backbone, backbone_id=backbone_id)
    model.load_state_dict(sd, strict=False)
    model = model.cuda()

    for f in os.listdir(df):
        img = torchvision.io.read_image(os.path.join(df, f)).float()/256
        if img.size()[1] < size[0] or img.size()[2] < size[1]:
            img = F.interpolate(img[None], size, mode='bilinear').squeeze(0)

        img = transforms.functional.center_crop(img, size)

        out = model(img.cuda()[None]) 
        save_mask(img.cpu().permute(1,2,0), out.argmax(dim=1)[0].float().cpu(), f'test_image_out/{f}_trained.png')


def main():
    backbone_path="fastervit_checkpoints/fastervit_0_224_1k.pth.tar"
    
    VPT = True
    VPT_PROMPT_LENGTH = 10
    backbone = faster_vit_factory(pretrained=True, vpt=VPT, vpt_prompt_length=VPT_PROMPT_LENGTH,model_path = backbone_path,freeze=True)

    BATCH_SIZE = 8
    EPOCHS = 60
    LR = 6e-4
    IMG_DIM = (224, 224)
    NUM_CLASSES = 2
    BACKBONE_ID='fastervit'
    DATASETS = ['recon', 'sacson','kitti','asrl']

    train(backbone,BATCH_SIZE,EPOCHS,LR,IMG_DIM,NUM_CLASSES, DATASETS, BACKBONE_ID)

def test():
    VPT = True
    VPT_PROMPT_LENGTH = 10
    backbone = faster_vit_factory(pretrained=False, vpt=VPT, vpt_prompt_length=VPT_PROMPT_LENGTH)
    if VPT:
        backbone = adjust_backbone_bias(backbone, VPT_PROMPT_LENGTH)
    
    IMG_DIM = (224, 224)

    checkpoint = 'trav_checkpoints/fastervit-epoch=59-step=125880-b=8-lr=6e-04.ckpt'
    test_images(backbone, checkpoint, IMG_DIM, 'test_images', 'fastervit')


if __name__=="__main__":
    test()
   # main()
