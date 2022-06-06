# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ..refer_nets.multimodal_fusion import VL_Concat,VL_Dynamic
from ..refer_nets.lang_encoder import RNNEncoder
# from operator import methodcaller
from ..builder import build_loss

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)

        lang_encoder_cfg=bbox_head['lang_encoder']
        lang_encoder_type = lang_encoder_cfg['type']
        if lang_encoder_type=='RNNEncoder':
            self.lang_encoder = RNNEncoder(vocab_size=lang_encoder_cfg['vocab_size'],
                                              word_embedding_size=lang_encoder_cfg['word_embedding_size'],
                                              word_vec_size=lang_encoder_cfg['word_vec_size'],
                                              hidden_size=lang_encoder_cfg['hidden_size'],
                                              bidirectional=lang_encoder_cfg['bidirectional'],
                                              input_dropout_p=lang_encoder_cfg['word_drop_out'],
                                              dropout_p=lang_encoder_cfg['rnn_drop_out'],
                                              n_layers=lang_encoder_cfg['rnn_num_layers'],
                                              rnn_type=lang_encoder_cfg['rnn_type'],
                                              variable_lengths=lang_encoder_cfg['variable_lengths'] > 0)
        multimodal_fusion_cfg = bbox_head['multimodal_fusion']
        multimodal_fusion_type = multimodal_fusion_cfg['type']
        if multimodal_fusion_type=='VL_Concat':
            self.multimodal_fusion = VL_Concat(hidden_size=multimodal_fusion_cfg['hidden_size'],
                num_input_channels=multimodal_fusion_cfg['num_input_channels'],
                num_output_channels=multimodal_fusion_cfg['num_output_channels'],
                num_featmaps=multimodal_fusion_cfg['num_featmaps'])
        if multimodal_fusion_type=='VL_Dynamic':
            self.multimodal_fusion = VL_Dynamic(hidden_size=multimodal_fusion_cfg['hidden_size'],
                num_input_channels=multimodal_fusion_cfg['num_input_channels'],
                num_output_channels=multimodal_fusion_cfg['num_output_channels'],
                num_featmaps=multimodal_fusion_cfg['num_featmaps'])

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        bbox_head.pop('lang_encoder')
        bbox_head.pop('multimodal_fusion')
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      att_labels, #refer
                      att_label_weights,
                      gt_bboxes_ignore=None,
                      refer_labels=None #refer
                      ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        refer_labels = [i.cuda(device=img.device) for i in refer_labels]
        context, hidden, embedded = self.lang_encoder(refer_labels)

        x = self.extract_feat(img)
        multimodal_feat=self.multimodal_fusion(x,context,hidden,embedded)

        losses = self.bbox_head.forward_train(multimodal_feat, img_metas, gt_bboxes,
                                              gt_labels,
                                              gt_bboxes_ignore)
        return losses


    def simple_test(self, img, img_metas, rescale=None,refer_labels=None):

        refer_labels = [i.cuda(device=img.device) for i in refer_labels[0]]
        """Test function without test time augmentation

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """

        context, hidden, embedded = self.lang_encoder(refer_labels)
        x = self.extract_feat(img)
        multimodal_feat = self.multimodal_fusion(x, context, hidden, embedded)
        results_list = self.bbox_head.simple_test(
            multimodal_feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels