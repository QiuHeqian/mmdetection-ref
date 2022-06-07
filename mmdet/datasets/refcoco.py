# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import json
import h5py
from .custom import CustomDataset
# from .registry import DATASETS
from .builder import DATASETS
import spacy
from collections import OrderedDict
import mmcv
from mmcv.utils import print_log
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import math



def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

@DATASETS.register_module
class RefCocoDataset(CustomDataset):
    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    CLASSES = ('object',)

    def load_annotations(self, ann_file):
        print('Loader loading data.json:', ann_file)
        self.info = json.load(open(ann_file))
        self.Images = self.info['Imgs']
        self.Anns = self.info['Anns']
        self.Refs = self.info['Refs']
        self.Sentences = self.info['Sents']
        self.Cats=self.info['Cats']
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.Cats)}
        if 'att_to_cnt' in self.info:
            self.AttToCnt = self.info['att_to_cnt']
        if 'att_to_ix' in self.info:
            self.AttToIx=self.info['att_to_ix']
            self.num_atts = len(self.AttToCnt)  # refer
        self.sent_infos=[]
        for id, info in self.Sentences.items():

            img_id=info['image_id']
            info['filename'] = self.Images[img_id]['file_name']
            info['width']=self.Images[img_id]['width']
            info['height']=self.Images[img_id]['height']
            self.sent_infos.append(info)
        print('we have %s images.' % len(self.Images))
        print('we have %s anns.' % len(self.Anns))
        print('we have %s refs.' % len(self.Refs))
        print('we have %s sentences.' % len(self.Sentences))
        self.label_length = self.info['label_length']
        print('label_length is', self.label_length)
        self.vocab_size = self.info['vocab_size']
        print('vocab size is', self.vocab_size)
        return self.sent_infos

    def get_ann_info(self, idx):
        sent_id = str(self.sent_infos[idx]['sent_id'])

        return self._parse_ann_info(sent_id)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds


    def _parse_ann_info(self, sent_id):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """

        sent_info=self.Sentences[sent_id]
        use_multiclass=False
        if use_multiclass:
            ref_category_id=sent_info['category_id']
        else:
            ref_category_id=0
        gt_ref_bboxes = xywh_to_xyxy(np.vstack([sent_info['bbox']])).astype(
            np.float32)
        gt_ref_masks = None

        gt_ref_labels = np.vstack([self.Sentences[sent_id]['h5_id_seq']])
        gt_bbox_labels = np.array([ref_category_id])

        if 'att_wds' in self.Sentences[sent_id]:
            att_wds=self.Sentences[sent_id]['att_wds']
            att_labels = np.ones((1,self.num_atts))
            att_label_weights = np.zeros((1,self.num_atts))
            att_to_ix=self.AttToIx
            att_to_cnt=self.AttToCnt

            if len(att_wds)>0:
                # scale = 10
                att_type_exist=[]
                for wd in att_wds:
                    att_type=wd.split('-')[0]
                    if att_type not in att_type_exist:
                        att_type_exist.append(att_type)
                    att_labels[0, self.AttToIx[wd]]=0
                for att_wd, index in att_to_ix.items():
                    if att_wd.split('-')[0] in att_type_exist:
                        num_att_wd = att_to_cnt[att_wd]
                        att_label_weights[0, index] = 1/math.sqrt(num_att_wd)
        else:
            att_labels=np.array([])
            att_label_weights =np.array([])
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        ann = dict(
            bboxes=gt_ref_bboxes,
            labels=gt_bbox_labels,
            refer_labels=gt_ref_labels,
            att_labels=att_labels,
            att_label_weights=att_label_weights,
            bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_ref_masks,
            roi_boxes=gt_ref_masks)

        return ann

    def evaluate(self,
                 results,
                 metric='Top1Acc',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=0.5,
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['Top1Acc']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        eval_results = OrderedDict()
        result_dict={}
        iou_thr_list=np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        correct_npy=np.zeros(len(iou_thr_list))

        for idx in range(len(self)):
            sent_id = str(self.sent_infos[idx]['sent_id'])
            bbox=self.Sentences[sent_id]['bbox']
            gt_bbox = xywh_to_xyxy(np.vstack([bbox]))
            image_id=self.Sentences[sent_id]['image_id']
            file_name = self.Images[image_id]['file_name']
            sentence=self.Sentences[sent_id]['sentence']
            result = results[idx]
            result_dict[sent_id]={}

            result_dict[sent_id]['pred_bbox']=[]
            result_dict[sent_id]['gt_bbox'] = bbox
            result_dict[sent_id]['image_id']=image_id
            result_dict[sent_id]['file_name'] = file_name
            result_dict[sent_id]['sentence']=sentence
            try:
                result_dict[sent_id]['pred_bbox'] = (xyxy_to_xywh(result)[0]).tolist()
                iou_box = bbox_overlaps(result[:,:4], gt_bbox)[0]
            except:
                iou_box=0
            for k in range(len(iou_thr_list)):
                iou_thr=iou_thr_list[k]
                if iou_box >= iou_thr:
                    correct_npy[k] = correct_npy[k] + 1

        det_acc_npy= correct_npy/len(self) * 100.0
        mean_det_acc = np.mean(det_acc_npy)
        print ('\n evaluating referring expression, det acc_50=',det_acc_npy[0])
        print('\n evaluating referring expression, det acc_75=', det_acc_npy[5])
        print('\n evaluating referring expression, det acc_mean=', mean_det_acc)
        msg = f'\n evaluating referring expression, det acc 50={det_acc_npy[0]}' \
              f'\n evaluating referring expression, det acc 75={det_acc_npy[5]}' \
              f'\nevaluating referring expression, det acc mean={mean_det_acc}'
        print_log(msg, logger=logger)
        return eval_results
