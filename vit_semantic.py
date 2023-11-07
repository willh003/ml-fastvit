import torch
import models
import os
from models.modules.mobileone import reparameterize_model
from models.fastvit import *
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from fpn.fpn import PanopticFPN
from PIL import Image
from pathlib import Path

from torchvision.models.detection import maskrcnn_resnet50_fpn
from utils import save_mask

from lightning.pytorch.loggers import TensorBoardLogger

"""
PLAN
    - fast vit + linear (or fpn, as described in fast vit paper)
    - freeze fast vit backbone, only train linear layer 
         - Interpretable: essentially learning the clip embeddings for traversability prompts, 
            projected into this transformer's space
    - use vpt online?? somehow to update during robot running

QUESTIONS:
    - Is fpn really necessary? Seems like it does same thing as ViT?
        - If so, how would fpn work on a vit? Is each embed dimension a separate channel?
        - If not, is a linear layer from embedding to class enough? (should be, this is basically clip)
"""
class VitSemantic(pl.LightningModule):
    def __init__(self, num_classes, img_dim, backbone=None,backbone_id='fastvit', lr=1e-3):
        """
        backbone: a ViT backbone returning a set of embeddings at different stages of convolution/attention
        embed_dims: dimension of embeddings output by the backbone, for use with the fpn head
        """
        super().__init__()
        self.num_classes = num_classes

        self.backbone = backbone
        if backbone_id == 'fastervit':
            embed_dims = [64, 128, 256, 512]
        elif backbone_id=='fastvit':
            embed_dims = [76, 152, 304, 608]
        else:
            raise Exception('Unimplemented backbone')

        self.head = self.fpn_head(embed_dims, img_dim[0], img_dim[1])
        self.img_dim = img_dim
        self.lr = lr

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if hasattr(self.backbone, 'trainable_params'):
            params =  list(self.head.parameters()) + self.backbone.trainable_params
        else:
            params = self.head.parameters()
        optim = torch.optim.AdamW(params, lr=self.lr)

        return optim

    def forward(self, img):
        embeddings = self.backbone(img)
        pred = self.head(embeddings)
        pred_interp = F.interpolate(pred, size=self.img_dim, mode='bilinear')
        return pred_interp

    def training_step(self, batch, batch_idx):
        src, trg = batch

        # backbone is frozen, so disable gradients
        with torch.no_grad():
            embeddings = self.backbone(src)
        
        probs = self.head(embeddings)
        probs_interp = F.interpolate(probs, size=self.img_dim, mode='bilinear')
        #NOTE: should loss be computed before or after image interpolation
        loss = F.cross_entropy(probs_interp, trg)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        src, trg = batch

        probs = self.forward(src)
        loss = F.cross_entropy(probs, trg)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        x = [p for p in self.head.parameters()]

        self.log('mean first param', x[0].mean(), prog_bar=True)

        # avg_loss = torch.tensor(self.training_step_outputs).mean()
        # self.log("train_loss", avg_loss, prog_bar=True)
        # self.training_step_outputs = []

    # def on_validation_epoch_end(self):
    #     avg_loss = torch.tensor(self.validation_step_outputs).mean()
    #     self.log("val_loss", avg_loss, prog_bar=True)
    #     self.validation_step_outputs = []


    def linear_head(self, embed_dim):
        """
        Returns a linear prediction layer to apply to the ViT embeddings
        Equivalent to learning the language prompt in Fast-ViT space that matches "traversable" in CLIP space
        """
        model = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, self.num_classes)
        )
        return model

    def fpn_head(self, embed_dims, img_height, img_width):
        """
        Returns an FPN-based head, according to 
        Kirillov, et al: Panoptic Feature Pyramid Networks https://arxiv.org/abs/1901.02446 
        PanopticFPN class adopted from pytorch-fpn: https://github.com/AdeelH/pytorch-fpn  
        """

        fmap_dims = []
        for i, dim in enumerate(embed_dims):

            w = round(img_width/(2**(i+2)))
            h = round(img_height/(2**(i+2)))
            fmap_dims.append((1, dim, h, w))

        fpn = PanopticFPN(in_feats_shapes=fmap_dims, hidden_channels=128)
        
        return fpn


    def get_boxes(self, masks):
        """
        ** Used for mask-rcnn only
        Inputs:
            masks: (B, C, H, W): segmentation probabilities for B images in batch and C classes
        Returns:
            boxes: (B, C, 4): minimum bounding box for each mask
            labels: (B, C): labels corresponding to each box
        """
        pass


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

def inference(model):
    # For inference
    model.eval()      
    model_inf = reparameterize_model(model)

def train(backbone, batch_size, epochs, lr, img_dim, num_classes, datasets=['recon'], backbone_id = 'fastvit'):
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


    model = VitSemantic(backbone=backbone, num_classes = num_classes, img_dim=img_dim,backbone_id=backbone_id, lr = lr)
    checkpoint_file=  backbone_id + '-{epoch}-{step}' + f'-b={batch_size}-lr={lr:.0e}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='checkpoints/',filename=checkpoint_file, every_n_epochs=5, monitor ='train_loss')



    logger = TensorBoardLogger("tb_logs", name=backbone_id)
    trainer = pl.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def test_images(backbone, checkpoint, size, df, backbone_id='fastvit'):
    h, w = size

    sd = torch.load(checkpoint)['state_dict']
    model = VitSemantic(num_classes=2, img_dim=(h, w), backbone=backbone, backbone_id=backbone_id)
    model.load_state_dict(sd, strict=False)
    model = model.cuda()

    #model2 = VitSemantic(2, (h, w), backbone).cuda()
    for f in os.listdir(df):
        img = torchvision.io.read_image(os.path.join(df, f)).float()/256
        if img.size()[1] < size[0] or img.size()[2] < size[1]:
            img = F.interpolate(img[None], size, mode='bilinear').squeeze(0)

        img = transforms.functional.center_crop(img, size)

        out = model(img.cuda()[None]) 
        #out2 = model2(img.cuda()[None]) 
        save_mask(img.cpu().permute(1,2,0), out.argmax(dim=1)[0].float().cpu(), f'test_image_out/{f}_trained.png')
        #save_mask(img.cpu().permute(1,2,0), out2.argmax(dim=1)[0].float().cpu(), f'test_image_out/{f}_untrained.png')
        #torchvision.utils.save_image(out.argmax(dim=1).float(), 'test_trained.png')
        #torch.tensor([F.interpolate(p, size=(2,h,w)) for p in pred])

def main():
    BATCH_SIZE = 16
    EPOCHS = 100
    LR = 1e-2
    IMG_DIM = (224, 224)
    NUM_CLASSES = 2


    backbone = fastvit_ma36(fork_feat=True)
    checkpoint = torch.load('/home/pcgta/Documents/playground/ml-fastvit/pretrained_checkpoints/fastvit_ma36.pth.tar')
    backbone.load_state_dict(checkpoint['state_dict'])
    DATASETS = ['recon', 'sacson','kitti','asrl']
    train(backbone = backbone,batch_size=BATCH_SIZE, epochs=EPOCHS, lr = LR, num_classes=NUM_CLASSES, img_dim=IMG_DIM, datasets=DATASETS)

    # backbone = fastvit_ma36(fork_feat=True)
    # backbone_params = torch.load('/home/pcgta/Documents/playground/ml-fastvit/pretrained_checkpoints/fastvit_ma36.pth.tar')
    # backbone.load_state_dict(backbone_params['state_dict'])

    # fastvit_checkpoint = '/home/pcgta/Documents/playground/ml-fastvit/checkpoints/fastvit-epoch=44-step=9360.ckpt'
    # test_images(backbone, fastvit_checkpoint, IMG_DIM, 'test_images', 'fastvit')

if __name__=='__main__':
    main()