# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList


class FeatureExtractorFromBoxes(torch.nn.Module):
    """
    Uses a GeneralizedRCNN model to re-compute the
    image features and extracts out the roi-aligned/pooled
    from the ground truth boxes
    """

    def __init__(self, grcnn_model):
        super().__init__()
        self.mdl = grcnn_model

    def forward(self, images, gtboxes):
        features = self.mdl.backbone(images.tensors)
        x, result, detector_losses = self.mdl.roi_heads(
            features, gtboxes, None)
        return x, result



class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image
class Predictor(object):
    # COCO categories for pretty print
    #CATEGORIES = ['short sleeve top','long sleeve top','short sleeve outwear','long sleeve outwear','vest','sling',
	#		'shorts','trousers','skirt','short sleeve dress','long sleeve dress','vest dress','sling dress']

    def __init__(
        self,
        model,
        CATEGORIES,
        dataset,
        confidence_threshold=0.5,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        if model == 'faster':
            config_file = "faster-retina/configs/e2e_faster_rcnn_R_50_FPN_1x_{}_test.yaml".format(dataset)
        if model == 'retinanet':
            config_file = 'faster-retina/configs/retinanet_R-50-FPN_1x-{}.yaml'.format(dataset)
        
        if model == 'maskrcnn':
            config_file = 'faster-retina/configs/e2e_mask_rcnn_R_50_FPN_1x-{}.yaml'.format(dataset)
        cfg.merge_from_file(config_file)
        self.cfg = cfg.clone()
        self.CATEGORIES = CATEGORIES
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device('cuda')
        self.model.to(self.device)
        self.min_image_size = min_image_size
        self.feat_extractor = FeatureExtractorFromBoxes(self.model)
        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def compute_features_from_bbox(self, original_image, gt_boxes):
        """
        Extracts features given the ground-truth boxes
        assume ground-truth boxes are list of boxes in xyxy format
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV
        Returns:
            features (BoxList): the ground truth boxes with features
            accessible using features.get_field()
        """
        # Convert gt boxes to BoxList
        gt_box_list = BoxList(
            gt_boxes, (original_image.shape[1], original_image.shape[0]), mode='xyxy').to(self.device)
        # Convert image as in `run_on_opencv_image`
        image = self.transforms(original_image)
        # Convert gt boxes for a single image to a list
        #print(image.size(1))
        gt_box_list = [gt_box_list.resize((image.size(2), image.size(1)))]
        image_list = to_image_list(
            image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            features = self.feat_extractor(image_list, gt_box_list)
        #print(features)
        # sanity check
        #assert len(features) == len(gt_box_list[0].bbox)
        #feats = gt_box_list[0]
        #feats.add_field('features', features)
        return features[0].cpu().detach().numpy()[0]

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image


    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image


    def get_detections(self,img):
        predictions = self.compute_prediction(img)
        top_predictions = self.select_top_predictions(predictions)

        labels = top_predictions.get_field('labels').tolist()
        bboxes = top_predictions.bbox.tolist()
        scores = top_predictions.get_field("scores").tolist()
        
        detections = []
        for bbox, score,label in zip(bboxes,scores,labels):
            detection =  bbox +  [score,label - 1]  #we won't have background
            detections.append(detection)

        return detections 