# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ..refer_nets.multimodal_fusion import VL_Concat,VL_Dynamic
from ..refer_nets.lang_encoder import RNNEncoder
# from operator import methodcaller
from ..builder import build_loss

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        lang_encoder_cfg = roi_head['lang_encoder']
        lang_encoder_type = lang_encoder_cfg['type']
        if lang_encoder_type == 'RNNEncoder':
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
        multimodal_fusion_cfg = roi_head['multimodal_fusion']
        multimodal_fusion_type = multimodal_fusion_cfg['type']
        if multimodal_fusion_type == 'VL_Concat':
            self.multimodal_fusion = VL_Concat(hidden_size=multimodal_fusion_cfg['hidden_size'],
                                               num_input_channels=multimodal_fusion_cfg['num_input_channels'],
                                               num_output_channels=multimodal_fusion_cfg['num_output_channels'],
                                               num_featmaps=multimodal_fusion_cfg['num_featmaps'])
        if multimodal_fusion_type == 'VL_Dynamic':
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

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pop('lang_encoder')
            roi_head.pop('multimodal_fusion')
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

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
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      att_labels, #refer
                      att_label_weights,
                      gt_bboxes_ignore=None,
                      refer_labels=None, #refer
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        refer_labels = [i.cuda(device=img.device) for i in refer_labels]
        context, hidden, embedded = self.lang_encoder(refer_labels)
        x = self.extract_feat(img)
        multimodal_feat = self.multimodal_fusion(x, context, hidden, embedded)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                multimodal_feat,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(multimodal_feat, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, refer_labels=None):
        """Test without augmentation."""
        refer_labels = [i.cuda(device=img.device) for i in refer_labels[0]]

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        multimodal_feat = self.multimodal_fusion(x, context, hidden, embedded)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(multimodal_feat, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            multimodal_feat, proposal_list, img_metas, rescale=rescale)[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
