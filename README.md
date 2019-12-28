# Clothing detection using YOLOv3, RetinaNet, Faster RCNN in ModaNet and DeepFashion2 datasets.

## Datasets

- DeepFashion2 dataset: https://github.com/switchablenorms/DeepFashion2 

- ModaNet dataset: https://github.com/eBay/modanet

## Models

- Faster RCNN, RetinaNet and Mask RCNN (only detection) trained with maskrcnn-benchmark https://github.com/facebookresearch/maskrcnn-benchmark/. To use this models please follow INSTALL instruccions in that repo and do the setup in the root folder of this repo. Not neccessary to use pytorch-nightly, you can use pytorch 1.2 instead.

- YOLOv3 trained with Darknet framework: https://github.com/AlexeyAB/darknet

- TridenNet trained with simpledet framework https://github.com/TuSimple/simpledet

- To do inference use a pytorch implementation of YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3.

- All the models trained with Resnet50 backbone, except YOLOv3 with Darknet53

## Weights

All weights and config files are in https://drive.google.com/drive/folders/1jXZZc5pp2OJCtmQYelzDgPzyuraAdxXP?usp=sharing

## Using

- Use <code>new_image_demo.py</code> , and choose dataset, and model. 
- Use <code>YOLOv3Predictor</code> class for YOLOv3 and <code>Predictor</code> class for Faster and RetinaNet and Mask.

## Coming soon

 - Update use of retrieval.
