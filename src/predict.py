#!/usr/bin/python3
""" Predict
Usage:
    predict.py [options] <img> <model>

Options:
    -h --help               Show this message
    -s --show               Show the detections in the image
"""
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from docopt import docopt
from utils import data
import matplotlib.pyplot as plt

class DetectionLayer(tf.keras.layers.Layer):
    def __init__( 
            self, 
            anchors, 
            nms_iou_thresh = 0.1,
            confidence_thresh = 0.5,
            max_detections_per_class  = 50,
            max_detections = 50,
            **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.anchors = anchors
        self.confidence_thresh = confidence_thresh
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections
        self.nms_iou_thresh = nms_iou_thresh

    def _decode_predictions(self, pred_boxes):
        anchors = self.anchors[None, ...]
        xy = pred_boxes[..., 0:2]*anchors[..., 2:] + anchors[..., :2]
        wh = tf.math.exp(pred_boxes[..., 2:]) * anchors[..., 2:]
        box_xywh = tf.concat([xy, wh], axis=-1)
        box_corners = data.convert_to_corners(box_xywh)
        return box_corners
        

    def call(self, predictions):
        pred_boxes = predictions[..., 0:4]
        cls = tf.nn.sigmoid(predictions[..., 4:])
        decoded_boxes = self._decode_predictions(pred_boxes)
        return tf.image.combined_non_max_suppression(
                    tf.expand_dims(decoded_boxes, axis=2),
                    cls,
                    self.max_detections_per_class, 
                    self.max_detections,
                    self.nms_iou_thresh,
                    self.confidence_thresh,
                    clip_boxes = False
                )

def predict(args):
    IMG_PATH = args["<img>"]
    MODEL_PATH = args["<model>"]
    ANCHOR_GRID = [(32, 32), (16, 16), (8, 8)]
    ASPECT_RATIOS = (1, 2/3, 3/2)

    model = load_model(MODEL_PATH, compile=False)
    img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)

    NET_SIZE = tuple(model.input.shape[1:3])
    IMG_SIZE = img.shape[0:2]
    ANCHORS = data.generate_multiple_anchors(ANCHOR_GRID, NET_SIZE, ASPECT_RATIOS)

    net_img = tf.image.resize(img/255, NET_SIZE)
    net_img = tf.constant(net_img, "float32")[None, ...]

    pred_boxes = model.predict(net_img)
    pred_boxes = DetectionLayer(ANCHORS, confidence_thresh=0.5)(pred_boxes)
    location_boxes = pred_boxes[0][0]
    classes = tf.cast(pred_boxes[2], tf.float32)
    classes = tf.transpose(classes)
    final_boxes = tf.concat([location_boxes, classes], axis=1)

    if args["--show"]:
        img1 = tf.cast(net_img[0]*255, tf.uint8)
        q = data.bndboxes_draw(img1, final_boxes)
        plt.imshow(q); plt.show()

if __name__ == "__main__":
    args = docopt(__doc__)
    predict(args)
    pass
