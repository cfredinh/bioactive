import os

from .bases import *
from torch.cuda.amp import autocast
from torch.profiler import profile, record_function, ProfilerActivity


class Identity(nn.Module):
    """An identity function."""
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class Classifier(BaseModel):
    """A wrapper class that provides different CNN backbones.
    
    Is not intended to be used standalone. Called using the DefaultWrapper class.
    """
    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)     

        if hasattr(cnn_models, self.backbone_type):
            incargs = {"aux_logits":False, "transform_input":False} if self.backbone_type == "inception_v3" else {}
            if self.pretrained:
                weight_pretrained = "IMAGENET1K_V1"
                print("using pretrained weights, ", weight_pretrained, " for ", self.backbone_type)
            else:
                weight_pretrained = None
            self.backbone = cnn_models.__dict__[self.backbone_type](weights=weight_pretrained, **incargs)
            fc_in_channels = self.backbone.fc.in_features
        else:
            raise NotImplementedError                
        self.backbone.fc = Identity()  # removing the fc layer from the backbone (which is manually added below)

        # modify stem and last layer
        self.fc = nn.Linear(fc_in_channels, self.n_classes)
        self.modify_first_layer(self.img_channels, self.pretrained)            
        
        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)   

    def forward(self, x, return_embedding=False):
        with autocast(self.use_mixed_precision):
            
            if self.freeze_backbone:
                self.backbone.eval()

            if isinstance(x, list) and hasattr(cnn_models, self.backbone_type):
                idx_crops = torch.cumsum(torch.unique_consecutive(
                    torch.tensor([inp.shape[-1] for inp in x]),
                    return_counts=True,
                )[1], 0)
                
                start_idx = 0
                for end_idx in idx_crops:
                    _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                    if start_idx == 0:
                        x_emb = _out
                    else:
                        x_emb = torch.cat((x_emb, _out))
                    start_idx = end_idx             
            else:
                x_emb = self.backbone(x)
                
            x = self.fc(x_emb)
            

            if return_embedding:
                return x, x_emb        
            else:
                return x
        
    def modify_first_layer(self, img_channels, pretrained):
        backbone_type = self.backbone.__class__.__name__
        if img_channels == 3:
            return

        if backbone_type == 'ResNet':
            conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                          'padding', 'dilation', "groups", "bias", "padding_mode"]
            conv1_defs = {attr: getattr(self.backbone.conv1, attr) for attr in conv_attrs}

            pretrained_weight = self.backbone.conv1.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.conv1 = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.conv1.weight.data = pretrained_weight 
            print(f"Adapting first chanel of network to {img_channels} channels")

        elif backbone_type == 'Inception3':
            conv_attrs = ['out_channels', 'kernel_size', 'stride', 
                          'padding', 'dilation', "groups", "bias", "padding_mode"]
            conv1_defs = {attr: getattr(self.backbone.Conv2d_1a_3x3.conv, attr) for attr in conv_attrs}

            pretrained_weight = self.backbone.Conv2d_1a_3x3.conv.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.Conv2d_1a_3x3.conv = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.Conv2d_1a_3x3.conv.weight.data = pretrained_weight                 
                
        else:
            raise NotImplementedError("channel modification is not implemented for {}".format(backbone_type))
