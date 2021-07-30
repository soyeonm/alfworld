import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
import torchvision.models as models



import os
import sys
import alfworld.gen.constants as constants


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

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



def get_model_instance_segmentation_different_backbone(num_classes, which_backbone):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    if which_backbone == 101:
        backbone = models.resnet101(pretrained=True)
    elif which_backbone == 152:
        backbone = models.resnet152(pretrained=True)
    backbone.out_channels = 2048 #same for 152

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
    mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
    

    model = MaskRCNN(
          backbone=backbone, 
          #num_classes=num_classes, 
          rpn_anchor_generator=anchor_generator, 
          rpn_head = model.rpn.head,
          box_predictor=model.roi_heads.box_predictor,
          mask_predictor=mask_predictor,
          box_roi_pool=roi_pooler #Just set everything else to model.attribute
          )
    #Let's hope this is fine?

    return model



def load_pretrained_model(path):
    mask_rcnn = get_model_instance_segmentation(len(constants.OBJECTS_DETECTOR)+1)
    mask_rcnn.load_state_dict(torch.load(path))
    return mask_rcnn