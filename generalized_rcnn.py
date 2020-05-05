# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.weakly_supervision_train = False
        self.roi_heads.weakly_supervision_train = False
        
    def open_weakly_supervision_train(self):
        self.weakly_supervision_train = True
        self.roi_heads.weakly_supervision_train = True
        self.roi_heads.training = False
        self.transform.training = False
        self.training = False
    
    def close_weakly_supervision_train(self):
        self.weakly_supervision_train = False
        self.roi_heads.weakly_supervision_train = False
        self.roi_heads.training = True
        self.transform.training = True
        self.training = False

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]               #记录原始图片大小，后面恢复预测box和mask需要用到
        images, targets = self.transform(images, targets)                       #标准化、剪裁图片
        #print("images size: {}".format(images.tensors.shape))
        features = self.backbone(images.tensors)                                #图片进过resnet_fpn网络得到特征图
        #for k,v in features.items():
        #    print(k)
        #    print(v.shape)
        #print("features size: {}".format(features))
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])                             #fpn网络可得到多个特征图，第一维图片数量，第二维通道数，第三第四维尺寸
        #print("features size: {}".format(features[0].shape))
        #print("features size: {}".format(features[1].shape))
        proposals, proposal_losses = self.rpn(images, features, targets)        #进过rpn网络，得到预选框Proposals
        #print("proposals size: {}".format(proposals[0].shape))
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)  #进过roi网络得到预测结果
        #print("detections size: {}".format(detections))
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)   #将预测结果还原回原图片尺寸大小
        #print("detections size: {}".format(detections))

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:                       #正常训练，只返回losses值
            return losses
        if self.weakly_supervision_train:                                             #弱监督学习方式，返回losses值和检测结果
            return losses, detections
        return detections