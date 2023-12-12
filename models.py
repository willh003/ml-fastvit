import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np



class CrossAttention(nn.Module):
    """
    Implements multi-head cross attention as described in Attention is All You Need
    Query as decoder dimension and key/value as encoder dimension
    """
    def __init__(self, d_enc, d_dec, d_model, p_drop=.1, h=8):

        super().__init__()

        d_proj = d_model // h
        self.d_proj = d_proj
        self.h = h

        self.q = nn.Linear(d_dec, d_proj * h)
        self.kv = nn.Linear(d_enc, 2*d_proj * h)
        self.w_multihead = nn.Linear(d_proj * h, d_model)
        self.dropout = nn.Dropout(p_drop)

        self.scale = (self.d_proj ** -.5)

    def forward(self, enc, dec):
        """
        enc: embedding representing encode signal (k, v)
        dec: embedding representing decode signal (q)
        """
        b_enc, n_enc, d_enc = enc.size()
        b_dec, n_dec, d_dec = dec.size()
        assert b_enc == b_dec, "batch size of src and trg do not match"

        # convert to following dimensions:
        # kv: (2, b, h, n_enc, d_proj)
        # q: (b, h, n_dec, d_proj)
        kv = self.kv(enc).view(b_enc, n_enc, 2, self.h, self.d_proj).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(dec).view(b_dec, n_dec, self.h, self.d_proj).permute(0, 2, 1, 3)

        # multi_attn: (b, h, n_dec, d_proj)
        multi_attn = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1) @ v

        # concatenate and project multihead
        # emb: (b, n_dec, d_model)
        emb = self.w_multihead(multi_attn.view(-1, n_dec, self.h * self.d_proj))

        emb = self.dropout(emb)
        return emb


class PatchEmbedding(nn.Module):
    """
    From https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html
    """
    def __init__(self, img_size=96, patch_size=16, embed_dim=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(embed_dim, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, x):
        # Output shape: (batch size, no. of channels, no. of patches)
        return self.conv(x).flatten(2).transpose(1,2)
    

class PriorFusionBackbone(nn.Module):
    """
    A backbone to fuse rgb image embeddings with a prior
        - Applies cross attention to image and prior embeddings, with the prior as the queries (encoded sequence)

    """
    def __init__(self, prior, encoder, prior_inp_dim, device='cuda', p_drop=.1, enable_backbone_grads = 'False'):

        """
        Inputs:
            prior: image prior model. Will be used as the transformer encoder
            encoder: image encoding model. Counterintuitively, will be used as the transformer decoder
            prior_inp_dim: input dimension of the prior (necessary for patch embedding)
            enable_backbone_grads: whether to enable grads for the prior/encoder, or just the cross attention
            
        this allows enc to attend to parts of the prior's signal, and lets the output have the same dimension as the encoder
        since the encoder is already trained to be good for downstream navigation, we suspect that keeping this dimensionality will improve performance
        """

        super(PriorFusionBackbone, self).__init__()

        self.prior_inp_dim = prior_inp_dim
        self.device = device
        p_drop = p_drop

        self.prior = prior
        self.enc = encoder

        # dimension of encoder embeddings (fixed for pretrained GNM)
        d_enc_embed = 1280
        n_enc_embed = 49

        # dimension of prior patch embeddings (tunable, since this is part of the model)
        d_prior_embed = 512 
        prior_patch_size = 16
        


        # d_model doesn't have to be d_enc_embed, but this lets us add a residual connection over the attention
        self.attn = CrossAttention(d_enc = d_prior_embed, d_dec=d_enc_embed, d_model=d_enc_embed)

        self.patch_embed = PatchEmbedding(
                            img_size=self.prior_inp_dim,
                            patch_size=prior_patch_size,
                            embed_dim = d_prior_embed)
        
        self.prior_pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, d_prior_embed))
        self.enc_pos_embed = nn.Parameter(torch.randn(1, n_enc_embed, d_enc_embed))
        
        self.prior_dropout = nn.Dropout(p=p_drop)
        self.enc_dropout = nn.Dropout(p=p_drop)
        self.ln = nn.LayerNorm(d_enc_embed)

        if not enable_backbone_grads:
            self.disable_backbone_grads()

        self.to(self.device)

    def forward(self, x):
        """
        x: (B, N, C, H, W)
        each point in x is a list of N recent images (6 for gnm)

        returns: vector of dimension (b, n_enc_patches, d_enc_embed)
            - this equals the dimension output by the encoder model
        """ 

        B, N, C, H, W = x.size()

        enc_embed = self.enc(x.view(B, N*C, H, W)) # encoding on all images, stacked along channel
        b_enc, d_enc, h_enc, w_enc = enc_embed.size()
        enc_embed = enc_embed.view(b_enc, d_enc, h_enc * w_enc).transpose(-1, -2)
        enc_embed += self.enc_pos_embed
        enc_embed = self.enc_dropout(enc_embed)

           
        # TODO: if using most recent, ensure that -1 really is the most recent image
        prior = self.prior(x[:,-1,:,:,:]) # prior on most recent image

        prior_embed = self.patch_embed(prior)
        prior_embed += self.prior_pos_embed
        prior_embed = self.prior_dropout(prior_embed)

        attn_out = self.attn(prior_embed, enc_embed)
        res_attn = attn_out + enc_embed
        ln_attn = self.ln(res_attn)

        return ln_attn

    def disable_backbone_grads(self):
        for p in self.prior.parameters():
            p.requires_grad_(False)

        for p in self.enc.parameters():
            p.requires_grad_(False)

class SegmentationModel(pl.LightningModule):
    """
    A lightning module for fine tuning semantic segmentation tasks
    """
    def __init__(self, model, img_dim, params=None, lr=1e-3):
        """        
        model: a segmentation model, returning class probabilities
        
        img_dim: tuple representing (h,w) of input images

        lr: initial learning rate
        
        params: specify the trainable params from model (defaults to self.params())
        """
        super().__init__()


        self.model = model

        self.img_dim = img_dim
        self.lr = lr

        self.training_step_outputs = []
        self.validation_step_outputs = []
        if params is None:
            self.trainable_params = [p for p in self.parameters() if p.requires_grad]
        else:
            self.trainable_params = params


    def configure_optimizers(self):

        optim = torch.optim.AdamW(self.trainable_params, lr=self.lr)

        return optim

    def forward(self, img):
        pred = self.model(img)

        #pred_interp = F.interpolate(pred, size=self.img_dim, mode='bilinear')
        return pred

    def training_step(self, batch, batch_idx):
        src, trg = batch
        probs = self.forward(src)
        #NOTE: should loss be computed before or after image interpolation

        loss = F.cross_entropy(probs, torch.argmax(trg, dim=1))
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        src, trg = batch

        probs = self.forward(src)
        
        loss = F.cross_entropy(probs, torch.argmax(trg, dim=1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        x = list(self.parameters())

        self.log('mean first param', x[0].mean(), logger=True) # log first param to ensure gradients are being calculated



class BC(pl.LightningModule):

    def __init__(self, backbone,  action_dim, continuous=False, class_weights=None, input_dim = (6, 3, 224, 224),feature_dim= (49, 1280),lr=3e-4):
        """
        Args:
            input_dim: the dimension of a single feature vector
            action_dim: the dimension of the action vector (1 for continuous command, 3 for l/r/0)
        """
        
        
        super().__init__()

        self.continuous = continuous

        self.input_dim = input_dim
        self.backbone = backbone
        self.head = self.create_head(feature_dim, action_dim, continuous=continuous)
        self.lr = lr
        self.class_weights = class_weights
        if continuous:
            self.loss = lambda pred, trg: F.huber_loss(pred, trg)
        else:
            self.loss = lambda pred, trg: F.cross_entropy(pred, trg, weight=self.class_weights)


    def forward(self, x):
        B, _, _, _, _ = x.size()
        x= x.view((B, *self.input_dim))

        with torch.no_grad():
            e = self.backbone(x)
        pred = self.head(e)

       
        return pred

    def training_step(self, batch, batch_idx):
        src, trg = batch

        pred = self.forward(src)

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
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        optim = torch.optim.AdamW(head_params + backbone_params, lr=self.lr)

        return optim

    def disable_grads(self):
        for param in self.parameters():
            param.requires_grad = False

    def create_head(self, in_dim, out_dim, continuous=False):

        *in_dim, = in_dim
        flattened_dim = np.prod(in_dim)

        if continuous:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_dim, 256),
                nn.Linear(256, out_dim) 
            )
        else:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_dim, 256),
                nn.Linear(256, out_dim), 
                nn.Softmax()   
            )
        

class FPNViTModule(nn.Module):
    def __init__(self, backbone, fpn):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn

    def forward(self, x):
        emb = self.backbone(x)
        out = self.fpn(emb)
        return out

def linear_head(self, embed_dim):
    """
    Returns a linear prediction layer to apply to the ViT embeddings
    Equivalent to learning the language prompt in Fast-ViT space that matches "traversable" in CLIP space
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(embed_dim, self.num_classes)
    )
    return model

def fpn_head(embed_dims, img_height, img_width):
    """
    Returns an FPN-based head, according to 
    Kirillov, et al: Panoptic Feature Pyramid Networks https://arxiv.org/abs/1901.02446 
    PanopticFPN class adopted from pytorch-fpn: https://github.com/AdeelH/pytorch-fpn  
    """
    from bc_trav.fpn.fpn import PanopticFPN

    fmap_dims = []
    for i, dim in enumerate(embed_dims):

        w = round(img_width/(2**(i+2)))
        h = round(img_height/(2**(i+2)))
        fmap_dims.append((1, dim, h, w))

    fpn = PanopticFPN(in_feats_shapes=fmap_dims, hidden_channels=128)
           

    return fpn