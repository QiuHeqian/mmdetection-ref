# MMDetection-Ref: An Open-Source Referring Grounding Toobox
<div align="center">
  <img src="resources/mmdetection-ref_framework.png" width="600"/>
</div>

## Introduction  
MMDetection-Ref is an open-source referring grounding toobox based on [MMDetection](https://github.com/open-mmlab/mmdetection), which ﬂexibly supports the integration of natural language and various visual detectors for end-to-end referring expression comprehension task.

## Installation
* MMDetection
* pytorch
* Please see [get_started.md](https://github.com/QiuHeqian/mmdetection-ref/blob/master/docs/get_started.md) for installation and the basic usage of MMDetection-Ref.

1. Clone the repository and then install it: 
``` 
git clone https://github.com/QiuHeqian/mmdetection-ref.git
cd mmdetection-ref  
pip install -v -e .  # or "python setup.py develop"  
```

## Data
## Train  
```
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in 'data/dataset/coco/'.
./tools/dist_train.sh configs/crossdet/crossdet_r50_fpn_1x_coco.py 8
```
```
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with VOC dataset in 'data/dataset/VOCdevkit/'.
./tools/dist_train.sh configs/crossdet/crossdet_r50_fpn_1x_voc.py 8
```

## Inference
```
./tools/dist_test.sh configs/crossdet/crossdet_r50_fpn_1x_coco.py work_dirs/crossdet_r50_fpn_1x_coco/epoch_12.pth 8 --eval bbox
```
## Acknowledgement
Thanks MMDetection team for the wonderful open source project!

## Citition
If you find CrossDet useful in your research, please consider citing:  
```
@inproceedings{qiu2021crossdet,  
  title={CrossDet: Crossline Representation for Object Detection},  
  author={Qiu, Heqian and Li, Hongliang and Wu, Qingbo and Cui, Jianhua and Song, Zichen and Wang, Lanxiao and Zhang, Minjian},  
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},  
  pages={3195--3204},  
  year={2021}  
}  
```
