import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
import torchvision.models as models

#from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.utils import load_state_dict_from_url

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import warnings

import os
import sys
import alfworld.gen.constants as constants




model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}


def _validate_trainable_layers(pretrained, trainable_backbone_layers, max_value, default_value):
    # dont freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(max_value))
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers

def load_maskrcnn_resnet101_or_152_fpn(pretrained, backbone_num, **kwargs):
    #default params
    progress=True; num_classes=91; pretrained_backbone=True; trainable_backbone_layers=5

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = True
    if backbone_num == 101:
        backbone = resnet_fpn_backbone('resnet101', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    elif backbone_num == 152:
        backbone = resnet_fpn_backbone('resnet152', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = MaskRCNN(backbone, num_classes, **kwargs)
    #if pretrained:
    #    state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
    #                                          progress=progress)
    #    model.load_state_dict(state_dict)
    #    overwrite_eps(model, 0.0)
    return model


def get_model_instance_segmentation(num_classes, backbone):
    # load an instance segmentation model pre-trained pre-trained on COCO
    if backbone == 50:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif backbone == 101 or backbone == 152:
        model = load_maskrcnn_resnet101_or_152_fpn(pretrained=True, backbone_num=backbone)


    anchor_generator = AnchorGenerator(
        sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
    model.rpn.anchor_generator = anchor_generator

    # 256 because that's the number of features that FPN returns
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



# num_classes = 2  # 1: person, 0: background
# in_features = model.roi_heads.box_predictor.cls_score.in_features

# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# backbone = torchvision.models.mobilenet_v2(pretrained=True).featres
# backbone.out_channels = 1280

# anchor_generator 
#   = AnchorGenerator(sizes=((32, 64, 128, 256, 512)), aspect_ratios=((0.5, 1.0, 2.0)))



# model = FasterRCNN(
#   backbone, 
#   num_classes=2, 
#   rpn_anchor_generator=anchor_generator, 
#   box_roi_pool=roi_pooler
#   )



# def get_model_instance_segmentation_different_backbone(num_classes, which_backbone):
#     # load an instance segmentation model pre-trained pre-trained on COCO
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#     if which_backbone == 101:
#         backbone = models.resnet101(pretrained=True)
#     elif which_backbone == 152:
#         backbone = models.resnet152(pretrained=True)
#     backbone.out_channels = 2048 #same for 152

#     anchor_generator = AnchorGenerator(
#         sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
#         aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
#     model.rpn.anchor_generator = anchor_generator

#     # 256 because that's the number of features that FPN returns
#     model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     # now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     # and replace the mask predictor with a new one
#     mask_predictor = MaskRCNNPredictor(in_features_mask,
#                                                        hidden_layer,
#                                                        num_classes)

#     roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
    

#     model = MaskRCNN(
#           backbone=backbone, 
#           #num_classes=num_classes, 
#           rpn_anchor_generator=anchor_generator, 
#           rpn_head = model.rpn.head,
#           box_predictor=model.roi_heads.box_predictor,
#           mask_predictor=mask_predictor,
#           box_roi_pool=roi_pooler #Just set everything else to model.attribute
#           )
#     #Let's hope this is fine?

#     return model



def load_pretrained_model(num_classes, path, backbone):
    mask_rcnn = get_model_instance_segmentation(num_classes, backbone)
    mask_rcnn.load_state_dict(torch.load(path))
    return mask_rcnn