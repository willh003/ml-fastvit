import torch
from bc_trav.faster_vit import adjust_backbone_bias
from bc_trav.gnm.gnm import GNM
from bc_trav.faster_vit import FasterVitBackbone
from pathlib import Path
from timm.models._builder import resolve_pretrained_cfg, _update_default_kwargs
from bc_trav.models import PriorFusionBackbone, SegmentationModel
import yaml

def ovt_factory(cfg_path):
    """
    Creates a navigation backbone that fuses the features of the GNM image encoder and a traversability prior
    """
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)

    trav_checkpoint_path = cfg['traversability']['checkpoint_path']
    vpt = cfg['traversability']['vpt']
    vpt_prompt_length = cfg['traversability']['vpt_prompt_length']
    trav_img_dim = tuple(cfg['traversability']['img_dim'])
    nav_checkpoint_path = cfg['navigation']['checkpoint_path']
    dropout = cfg['training']['dropout']
    full_fine_tune = cfg['training']['full_fine_tune']
    device = cfg['device']

    prior = trav_prior_factory(trav_checkpoint_path, img_dim=trav_img_dim, vpt=vpt, vpt_prompt_length=vpt_prompt_length, device=device)
    nav_enc = gnm_encoder_factory(nav_checkpoint_path)

    model = PriorFusionBackbone(prior=prior, encoder=nav_enc, prior_inp_dim=trav_img_dim, device=device, p_drop=dropout, enable_backbone_grads=full_fine_tune)
    return model

def gnm_encoder_factory(ckpt):
    """
    Creates a navigation network
    Uses the observation network of GNM (Dhruv, et al. General Navigation Models to Drive Any Robot): 
    https://github.com/robodhruv/visualnav-transformer
    """

    sd = torch.load(ckpt)
    gnm = GNM()
    obsnet = gnm.obs_mobilenet
    obsnet.load_state_dict(sd)
    return obsnet

def trav_prior_factory(ckpt, img_dim=(224, 224), vpt=True, vpt_prompt_length=10, device='cuda'):

    backbone = faster_vit_factory(pretrained=False, vpt=vpt, vpt_prompt_length=vpt_prompt_length)
    if vpt:
        backbone = adjust_backbone_bias(backbone, vpt_prompt_length)

    sd = torch.load(ckpt)['state_dict']
    trav = SegmentationModel(num_classes=2, img_dim=img_dim, backbone=backbone, backbone_id='fastervit')
    trav.load_state_dict(sd, strict=False)
    trav = trav.to(device)
    return trav

def faster_vit_factory(pretrained=False, **kwargs):
    """
    Initializes a new fastervit backbone
    
    If pretrained, loads the weights defined by kwarg "model_path"
    """
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