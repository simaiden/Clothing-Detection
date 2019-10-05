# Clothing detection using YOLOv3, RetinaNet, Faster RCNN in ModaNet and DeepFashion2 datasets.

## Datasets

- DeepFashion2 dataset: https://github.com/switchablenorms/DeepFashion2 

- ModaNet dataset: https://github.com/eBay/modanet

## Models

- Faster RCNN and RetinaNet trained with maskrcnn-benchmark https://github.com/facebookresearch/maskrcnn-benchmark/. To use this models please follow INSTALL instruccions in that repo and do the setup in the root folder of this repo. Not neccessary to use pytorch-nightly, you can use pytorch 1.2 instead.

- YOLOv3 trained with Darknet framework: https://github.com/AlexeyAB/darknet

- To do inference use a pytorch implementation of YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3.


## Weights

### DeepFashion2

- Faster RCNN https://drive.google.com/file/d/1DrHTPsZEZt_7NeQKgFuMUR1lJLAeJgTS/view?usp=sharing

- RetinaNet https://drive.google.com/file/d/1T7wp3yL-tLEtL-a9BWGfychmKV3rWDc-/view?usp=sharing

- YOLOv3: https://drive.google.com/file/d/14-_ctjNtIWLhhLsqwYQzPUNCzh4U1coq/view?usp=sharing

### ModaNet

- Faster RCNN https://drive.google.com/file/d/1pDLpXM7DtpaZTGdSjtxrmX-r9Tcs3wUX/view?usp=sharing

- RetinaNet https://drive.google.com/file/d/1WzW3wR7FoKFA9qgAGaKEUS4KEydmlwcp/view?usp=sharing

- YOLOv3 soon


## Using

- Use <code>new_image_demo.py</code> , and choose dataset, and model. 
- Use <code>YOLOv3Predictor</code> class for YOLOv3 and <code>Predictor</code> class for Faster and RetinaNet.

## Coming soon

 - Update use of retrieval.