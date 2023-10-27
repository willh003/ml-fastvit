import torch
from torch.autograd import Variable
from collections import deque
from fastervit.models.faster_vit import FasterViT, FasterViTLayer, window_partition, window_reverse
from fastervit.models.faster_vit import faster_vit_0_224
from timm.models._builder import resolve_pretrained_cfg, _update_default_kwargs
from pathlib import Path
from vit_semantic import train, test_images


class VPT_FasterViTLayer(FasterVitLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, **kwargs):
        """
        kwargs:
            soft_prompt: (C, H, W) = *x.size() \ B
        """
        ct = self.global_tokenizer(x) if self.do_gt else None
        B, C, H, W = x.shape

        if 'soft_prompt' in kwargs:
            vit_to_concat = self.soft_prompt.unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.concat((x, vit_to_concat), dim=1)
        if self.transformer_block:
            x = window_partition(x, self.window_size)
        for bn, blk in enumerate(self.blocks):
            x, ct = blk(x, ct)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, H, W, B)
        if self.downsample is None:
            return x
        return self.downsample(x)

class FasterVitBackbone(FasterViT):
    """
    Extension of FasterViT class, allowing for vpt on inputs and fpn on outputs
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        # might be levels[3] instead
        self.outputs = deque(maxlen=5)

        # Kind of weird, since this is fpn on the embeddings and each ViT layer output, but dimensions seem good
        # Maybe remove levels[2] because it is same size as levels[3]
        #[torch.Size([1, 64, 56, 56]), torch.Size([1, 128, 28, 28]), torch.Size([1, 256, 14, 14]), 
        # torch.Size([1, 512, 7, 7]), torch.Size([1, 512, 7, 7])]
        # self.patch_embed.register_forward_hook(self.hook)
        # self.levels[0].register_forward_hook(self.hook)
        # self.levels[1].register_forward_hook(self.hook)
        # #self.levels[2].register_forward_hook(self.hook)
        # self.levels[3].register_forward_hook(self.hook)

        # run once without vpt, to get embed dim
        self.vpt = False
        resolution = kwargs.get('resolution', 224)

        self.forward_features(torch.rand((1,3,resolution,resolution)))

        if 'vpt' in kwargs:
            self.vpt = kwargs.get('vpt')
            if self.vpt:
                self.vpt_prompt_length = kwargs.get('vpt_prompt_length')
                self.soft_prompt = self.get_vpt_params(self.layer2_size, self.vpt_prompt_length)


    def get_output_shape(self, model, image_dim):
        return model(torch.rand(*(image_dim))).data.shape

    def get_vpt_params(self, patch_dim, prompt_length):
        params = torch.nn.Parameter(data=torch.zeros(prompt_length, patch_dim[-2], patch_dim[-1], requires_grad=True))
        torch.nn.init.xavier_uniform_(params)    
        return params

    def forward_features(self, x):
        x = self.patch_embed(x)
        self.outputs.append(x)
        i=0
        for level in self.levels:
            print(i, level.conv)
            print(x.size())
            if i==2: # first transformer layer is at level 3 (first two layers are convs)
                if self.vpt: # shallow vpt on first transformer layer
                    b, _,_,_ = x.size()

                self.layer2_size =x.size() # store this, for vpt purposes
            
            x = level(x)
            if i!=2: # for fpn
                self.outputs.append(x)
            i+=1

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




def create_faster_vit(pretrained=False, **kwargs):
    print(kwargs)
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
    VPT_PROMPT_LENGTH = 1
    backbone = create_faster_vit(pretrained=True, vpt=VPT, vpt_prompt_length=VPT_PROMPT_LENGTH,model_path = backbone_path)
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 5e-4
    IMG_DIM = (224, 224)
    NUM_CLASSES = 2
    BACKBONE_ID='fastervit'
    DATASETS = ['recon', 'sacson','kitti','asrl']

    train(backbone,BATCH_SIZE,EPOCHS,LR,IMG_DIM,NUM_CLASSES, DATASETS, BACKBONE_ID)

def test():
    VPT = True
    VPT_PROMPT_LENGTH = 1
    backbone_path="/home/pcgta/Documents/playground/ml-fastvit/pretrained_checkpoints/fastervit_0_224_1k.pth.tar"
    backbone = create_faster_vit(pretrained=True, model_path = backbone_path, vpt=VPT, vpt_prompt_length=VPT_PROMPT_LENGTH)
    IMG_DIM = (224, 224)

    fastervit_checkpoint = 'checkpoints/fastervit-epoch=9-step=167840.ckpt'
    test_images(backbone, fastervit_checkpoint, IMG_DIM, 'test_images', 'fastervit')


if __name__=="__main__":
    test()
    #main()
