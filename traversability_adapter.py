import torch
import models
from timm.models import create_model
from models.modules.mobileone import reparameterize_model
from torchvision.models.detection import maskrcnn_resnet50_fpn
"""
PLAN
    - mask rcnn with fpn on top of fast vit (as described in fast vit paper)
    - freeze fast vit backbone, only train mask rcnn
    - use vpt online?? somehow to update during robot running

QUESTIONS:
    - Downresing: bilinear interpolation or striding?
    - 

"""
EMBED_DIM = 768
class FastVitSemantic(torch.nn.module):
    def __init__(self, num_classes):
        # To Train from scratch/fine-tuning
        self.backbone = create_model("fastvit_ma36")
        # Load unfused pre-trained checkpoint for fine-tuning
        # or for downstream task training like detection/segmentation
        checkpoint = torch.load('/home/pcgta/Documents/playground/ml-fastvit/checkpoints/fastvit_ma36_reparam.pth.tar')
        self.backbone.load_state_dict(checkpoint['state_dict'])
        self.head = self.get_head()
        
        
        
        self.optim = torch.optim.AdamW(lr=1e-3)
        self.num_classes = num_classes

    def get_head(self, embedding):
        model = torch.nn.Sequential(
            torch.nn.Linear(EMBED_DIM),
            torch.nn.Softmax()
        )
        #maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)

        return model
         

    def forward(self, img):
        embeddings = self.backbone(img)
        pred = self.head(embeddings)
        return pred

    def train(self, dataloader):
        self.head.train()

        for i, (images, masks) in enumerate(dataloader):
            boxes, labels = self.get_boxes(masks)

            # NOTE: not sure if this is the correct format for target
            # Also, images might have to be a list for the mask rcnn
            targets = [{"boxes":boxes[i], "labels":labels[i],"masks":masks[i] }]

            embeddings = self.backbone(images)
            losses = self.head(images=embeddings, targets=targets)
            # TODO: what does losses return? (it is some sort of dict)
            
            


    def get_boxes(self, masks):
        """
        Inputs:
            masks: (B, C, H, W): segmentation probabilities for B images in batch and C classes
        Returns:
            boxes: (B, C, 4): minimum bounding box for each mask
            labels: (B, C): labels corresponding to each box
        """
        pass




def inference(model):
    # For inference
    model.eval()      
    model_inf = reparameterize_model(model)