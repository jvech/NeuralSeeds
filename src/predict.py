#!/usr/bin/python3
""" Predict
Usage:
    predict.py [options] <img> <model>

Options:
    -h --help               Show this message
    -s --show               Show the detections in the image
"""

try: import silence_tensorflow.auto
except ModuleNotFoundError: pass
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from docopt import docopt
from utils import data
import matplotlib.pyplot as plt

def predict(args):
    IMG_PATH = args["<img>"]
    MODEL_PATH = args["<model>"]
    #ANCHOR_GRID = [(32, 32), (16, 16), (8, 8)]
    #ASPECT_RATIOS = (1, 2/3, 3/2)
    ANCHOR_GRID, ASPECT_RATIOS, NET_INPUT_SIZE = data.load_config("config.json")

    model = load_model(MODEL_PATH, compile=False)
    img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)

    #NET_SIZE = tuple(model.input.shape[1:3])
    ANCHORS = data.generate_multiple_anchors(ANCHOR_GRID, NET_INPUT_SIZE, ASPECT_RATIOS)

    net_img = tf.image.resize(img/255, NET_INPUT_SIZE)
    net_img = tf.constant(net_img, "float32")[None, ...]

    pred_boxes = model.predict(net_img)
    pred_boxes = data.NonMaximumSupression(ANCHORS, confidence_thresh=0.5)(pred_boxes)
    location_boxes = pred_boxes[0][0]
    classes = tf.cast(pred_boxes[2], tf.float32)
    classes = tf.transpose(classes)
    final_boxes = tf.concat([location_boxes, classes], axis=1)
    img1 = tf.cast(net_img[0]*255, tf.uint8)
    img1, final_boxes = data.img_pre(img1, final_boxes, img.shape[:2])
    final_boxes = tf.cast(final_boxes, tf.uint32)
    print("x\ty\tw\th\tcls")
    for x, y, w, h, cls in final_boxes:
        if w != 0 and h != 0:
            print("%d\t%d\t%d\t%d\t%d"%(x, y, w, h, cls))

    if args["--show"]:
        q = data.bndboxes_draw(img1, final_boxes)
        plt.imshow(q); 
        plt.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    args = docopt(__doc__)
    predict(args)
    pass
