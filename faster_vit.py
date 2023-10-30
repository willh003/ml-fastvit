import torch
from torch.autograd import Variable
from collections import deque
from fastervit.models.faster_vit import HAT, FasterViT, FasterViTLayer, window_partition, window_reverse, PatchEmbed, ct_window, ct_dewindow, TokenInitializer, ConvBlock, Downsample
from fastervit.models.faster_vit import faster_vit_0_224
from timm.models._builder import resolve_pretrained_cfg, _update_default_kwargs
from timm.models.layers import LayerNorm2d
from pathlib import Path
import torch.nn as nn
from vit_semantic import train, test_images


class VPT_FasterViTLayer(FasterViTLayer):
    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 num_heads,
                 window_size,
                 ct_size=1,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 only_local=False,
                 hierarchy=True,
                 do_propagation=False
                 ):
        """
        Args:
            dim: feature size dimension.
            depth: layer depth.
            input_resolution: input resolution.
            num_heads: number of attention head.
            window_size: window size.
            ct_size: spatial dimension of carrier token local window.
            conv: conv_based stage flag.
            downsample: downsample flag.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            layer_scale: layer scale coefficient.
            layer_scale_conv: conv layer scale coefficient.
            only_local: local attention flag.
            hierarchy: hierarchical attention flag.
            do_propagation: enable carrier token propagation.
        """
        super(FasterViTLayer, self).__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([
                ConvBlock(dim=dim,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layer_scale=layer_scale_conv)
                for i in range(depth)])
            self.transformer_block = False
        else:
            sr_ratio = input_resolution // window_size if not only_local else 1
            self.blocks = nn.ModuleList([
                VPT_HAT(dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    sr_ratio=sr_ratio,
                    window_size=window_size,
                    last=(i == depth-1),
                    layer_scale=layer_scale,
                    ct_size=ct_size,
                    do_propagation=do_propagation,
                    )
                for i in range(depth)])
            self.transformer_block = True
        self.downsample = None if not downsample else Downsample(dim=dim)
        if len(self.blocks) and not only_local and input_resolution // window_size > 1 and hierarchy and not self.conv:
            self.global_tokenizer = TokenInitializer(dim,
                                                     input_resolution,
                                                     window_size,
                                                     ct_size=ct_size)
            self.do_gt = True
        else:
            self.do_gt = False

        self.window_size = window_size
    
    def forward(self, x, **kwargs):
        """
        kwargs:
            soft_prompt: (N, C)
        """
        ct = self.global_tokenizer(x) if self.do_gt else None
        B, C, H, W = x.shape
        vit= False

        if self.transformer_block:
            x = window_partition(x, self.window_size)
            if 'soft_prompt' in kwargs:
                vit=True
                soft_prompt = kwargs.get('soft_prompt')
        for bn, blk in enumerate(self.blocks):
            if vit:
                x, ct = blk(x, ct, soft_prompt=soft_prompt)
            else:
                x, ct = blk(x, ct)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, H, W, B)
        if self.downsample is None:
            return x
        return self.downsample(x)


class VPT_HAT(HAT):
    """
    Hierarchical attention (HAT) based on: "Hatamizadeh et al.,
    FasterViT: Fast Vision Transformers with Hierarchical Attention
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x, carrier_tokens, **kwargs):

        B, T, N = x.shape
        
        vpt=False
        if 'soft_prompt' in kwargs:
            vpt=True
            soft_prompt = kwargs.get('soft_prompt') # soft prompt should be B, N_p, C 
            soft_prompt = soft_prompt.unsqueeze(0).expand(B, -1, -1)
            _, N_p, _ = soft_prompt.size()


        ct = carrier_tokens
        x = self.pos_embed(x)

        if self.sr_ratio > 1:
            # do hierarchical attention via carrier tokens
            # first do attention for carrier tokens
            Bg, Ng, Hg = ct.shape

            # ct are located quite differently
            ct = ct_dewindow(ct, self.cr_window*self.sr_ratio, self.cr_window*self.sr_ratio, self.cr_window)

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)

            # attention plus mlp
            ct = ct + self.hat_drop_path(self.gamma1*self.hat_attn(self.hat_norm1(ct)))
            ct = ct + self.hat_drop_path(self.gamma2*self.hat_mlp(self.hat_norm2(ct)))

            # ct are put back to windows
            ct = ct_window(ct, self.cr_window * self.sr_ratio, self.cr_window * self.sr_ratio, self.cr_window)

            ct = ct.reshape(x.shape[0], -1, N)
            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)

        if vpt:
            x = torch.cat((soft_prompt, x), dim=1)

        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3*self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4*self.mlp(self.norm2(x)))

        if vpt:
            _, x = x.split([N_p, x.shape[1] - N_p], dim=1) # remove soft prompt

        if self.sr_ratio > 1:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split([x.shape[1] - self.window_size*self.window_size, self.window_size*self.window_size], dim=1)
            ct = ctr.reshape(Bg, Ng, Hg) # reshape carrier tokens.
            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = ctr.transpose(1, 2).reshape(B, N, self.cr_window, self.cr_window)
                x = x + self.gamma1 * self.upsampler(ctr_image_space.to(dtype=torch.float32)).flatten(2).transpose(1, 2).to(dtype=x.dtype)
        return x, ct

class FasterVitBackbone(FasterViT):
    """
    Extension of FasterViT class, allowing for vpt on inputs and fpn on outputs
    """
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 ct_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 layer_norm_last=False,
                 hat=[False, False, True, False],
                 do_propagation=False,
                 **kwargs):

        #super().__init__(**kwargs)
        super(FasterViT, self).__init__()


        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        if hat is None: hat = [True, ]*len(depths)
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            layer_dim = int(dim * 2 ** i)
            
            level = VPT_FasterViTLayer(dim=layer_dim,
                                   depth=depths[i],
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   ct_size=ct_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   conv=conv,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                   downsample=(i < 3),
                                   layer_scale=layer_scale,
                                   layer_scale_conv=layer_scale_conv,
                                   input_resolution=int(2 ** (-2 - i) * resolution),
                                   only_local=not hat[i],
                                   do_propagation=do_propagation)
            self.levels.append(level)
        self.norm = LayerNorm2d(num_features) if layer_norm_last else nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        self.outputs = deque(maxlen=5)

        # run once without vpt, to get embed dim
        resolution = kwargs.get('resolution', 224)
        self.vpt = kwargs.get('vpt', False)
        freeze = kwargs.get('freeze', True) 

        if self.vpt:
            if freeze:
                self.disable_grads() # disable before adding soft prompt
            self.vpt_prompt_length = kwargs.get('vpt_prompt_length')
            l2_channels = 256
            l3_channels = 512
            self.l2_prompt = self.get_vpt_params(l2_channels, self.vpt_prompt_length)
            self.l3_prompt = self.get_vpt_params(l3_channels, self.vpt_prompt_length)
            self.trainable_params = [self.l2_prompt, self.l3_prompt]


    def disable_grads(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_vpt_params(self, channels, prompt_length):
        params = torch.nn.Parameter(data=torch.zeros(prompt_length, channels, requires_grad=True))
        torch.nn.init.xavier_uniform_(params)    
        return params

    def forward_features(self, x):
        x = self.patch_embed(x)
        self.outputs.append(x)
        for i, level in enumerate(self.levels):
            if i == 2 and self.vpt:
                x = level(x, soft_prompt=self.l2_prompt)
            elif i == 3 and self.vpt:
                x = level(x, soft_prompt=self.l3_prompt)
            else:
                x = level(x)
            if i !=2:
                self.outputs.append(x)
            

        x = self.norm(x)
        return x

    def forward(self, x):
        """
        Override forward method to only use head
        """

        self.outputs.clear()
        embed=self.forward_features(x)

        #x = self.forward_features(x)
        return list(self.outputs)
    
    def hook(self, model, input, output):
        self.outputs.append(output.detach())

def adjust_backbone_bias(backbone, add_seq_length):
    for i, level in enumerate(backbone.levels):
        if i == 2 or i == 3:
            for blk in level.blocks:
                b, c, h, w = blk.attn.pos_emb_funct.relative_bias.size()
                blk.attn.pos_emb_funct.relative_bias = torch.zeros((b, c, h+add_seq_length, w+add_seq_length))
    return backbone


def create_faster_vit(pretrained=False, **kwargs):
    depths = kwargs.pop("depths", [2, 3, 6, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [7, 7, 7, 7])
    ct_size = kwargs.pop("ct_size", 2)
    dim = kwargs.pop("dim", 64)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    model_path = kwargs.pop("model_path", "/tmp/faster_vit_0.pth.tar")
    hat = kwargs.pop("hat", [False, False, True, False])
    pretrained_cfg = resolve_pretrained_cfg('faster_vit_0_224').to_dict()
    _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter=None)
    model = FasterVitBackbone(depths=depths,
                      num_heads=num_heads,
                      window_size=window_size,
                      ct_size=ct_size,
                      dim=dim,
                      in_dim=in_dim,
                      mlp_ratio=mlp_ratio,
                      resolution=resolution,
                      drop_path_rate=drop_path_rate,
                      hat=hat,
                      **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model

def main():
    backbone_path="/home/pcgta/Documents/playground/ml-fastvit/pretrained_checkpoints/fastervit_0_224_1k.pth.tar"
    
    VPT = True
    VPT_PROMPT_LENGTH = 10
    backbone = create_faster_vit(pretrained=True, vpt=VPT, vpt_prompt_length=VPT_PROMPT_LENGTH,model_path = backbone_path,freeze=True)

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
    backbone_path="/home/pcgta/Documents/playground/ml-fastvit/pretrained_checkpoints/fastervit_0_224_1k.pth.tar"
    backbone = create_faster_vit(pretrained=True, model_path = backbone_path, vpt=VPT, vpt_prompt_length=VPT_PROMPT_LENGTH)
    if VPT:
        backbone = adjust_backbone_bias(backbone, VPT_PROMPT_LENGTH)
    
    IMG_DIM = (224, 224)

    fastervit_checkpoint = 'checkpoints/fastervit-epoch=59-step=125880-b=8-lr=6e-04.ckpt'
    test_images(backbone, fastervit_checkpoint, IMG_DIM, 'test_images', 'fastervit')


if __name__=="__main__":
    test()
   # main()
