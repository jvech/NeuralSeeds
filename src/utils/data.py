import os
from os import path
import cv2
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
#from utils.metrics import get_IoU

def get_IoU(box1: "tf.Tensor", box2: "tf.Tensor") -> float:
    """Return The intersection over Union over 2 corner boxes"""

    lu = tf.maximum(box1[:2], box2[:2])
    rd = tf.minimum(box1[2:], box2[2:])

    area1 = (box1[0] - box1[2]) * (box1[1] - box1[3])
    area2 = (box2[0] - box2[2]) * (box2[1] - box2[3])

    wh_intersection = tf.maximum(0.0, rd - lu)
    Intersection = tf.reduce_prod(wh_intersection)
    Union = tf.maximum(area1 + area2 - Intersection, 1e-8)
    return float(Intersection / Union)

def get_IoUs(boxes1: "tf.Tensor", boxes2: "tf.Tensor") -> "tf.Tensor":
    """Return a (M, N) shape Tensor with Ious from boxes1 and boxes2

    boxes1 and boxes2 have shapes (M, 4) and (N, 4) respectively and are
    described by their corners (xmin, ymin, xmax, ymax)
    """
    lu = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rd = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    area1 = (boxes1[:,0] - boxes1[:,2]) * (boxes1[:,1] - boxes1[:,3])
    area2 = (boxes2[:,0] - boxes2[:,2]) * (boxes2[:,1] - boxes2[:,3])
    wh_intersection = tf.maximum(0.0, rd - lu)
    Intersection = tf.reduce_prod(wh_intersection, axis=2)

    Union = tf.maximum(area1[:, None], area2 - Intersection, 1e-8)
    return tf.clip_by_value(Intersection / Union, 0.0, 1.0)

def convert_to_xyhw(box_corners):
    xmin = box_corners[..., 0] 
    ymin = box_corners[..., 1] 
    xmax = box_corners[..., 2] 
    ymax = box_corners[..., 3] 

    w = xmax - xmin
    h = ymax - ymin

    x = xmin + w/2
    y = ymin + h/2
    if int(box_corners.shape[-1]) > 4:
        classes = box_corners[..., 4]
        return tf.stack([x, y, w, h, classes], axis=-1)
    return tf.stack([x, y, w, h], axis=-1)

def convert_to_corners(box_xyhw):
    x = box_xyhw[..., 0]
    y = box_xyhw[..., 1]
    w = box_xyhw[..., 2]
    h = box_xyhw[..., 3]

    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2
    if int(box_xyhw.shape[-1]) > 4:
        classes = box_xyhw[..., 4]
        return tf.stack([xmin, ymin, xmax, ymax, classes], axis=-1)
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

def bndboxes_draw(img: "tf.Tensor", boxes: "tf.Tensor") -> "np.ndarray":
    """Plot boxes in corner format (xmin, ymin, xmax, ymax, cls)"""

    X = img.numpy().astype("uint8").copy()
    Y = boxes.numpy().astype("int").copy()
    for x1, y1, x2, y2, c in Y:
        pt1, pt2 = (x1, y1), (x2, y2)
        if c == 0.0: COLOR = (255, 0, 0)          #RED
        elif c == 1.0: COLOR = (255, 255, 0)      #YELLOW
        else: COLOR = (0, 0, 0)                 #BLACK
        X = cv2.rectangle(X, pt1, pt2, COLOR, 1)
    return tf.constant(X, dtype=tf.uint8)

def generate_anchors(featuremap_size: tuple, img_size: tuple, aspect_ratio: float=1):
    """Generate a (M x N x 4) grid of boxes for an image"""

    x_anchors = tf.range(featuremap_size[0], dtype=tf.float32) + 0.5
    y_anchors = tf.range(featuremap_size[1], dtype=tf.float32) + 0.5

    x_anchors *=  img_size[0] / featuremap_size[0]
    y_anchors *=  img_size[1] / featuremap_size[1]

    X, Y = tf.meshgrid(x_anchors, y_anchors)
    centers = tf.stack([X, Y], axis=-1)

    w =  (img_size[0] / featuremap_size[0]) * aspect_ratio
    h =  (img_size[1] / featuremap_size[1]) / aspect_ratio
    W = tf.tile(tf.constant([[w]], tf.float32), featuremap_size)
    H = tf.tile(tf.constant([[h]], tf.float32), featuremap_size)
    dims = tf.stack([W, H], axis=-1)

    xywh_anchors = tf.concat([centers, dims], axis=-1)
    return xywh_anchors

def generate_multiple_anchors(featuremap_sizes: list, img_size: tuple, aspect_ratios: tuple):
    """Return an array of anchor boxes of different featuremap shapes and aspect_ratios"""

    anchors = []
    for featmap_size in featuremap_sizes:
        for aspect_ratio in aspect_ratios:
            anchor = generate_anchors(featmap_size, img_size, aspect_ratio)
            anchor = tf.reshape(anchor, [-1, 4])
            anchors.append(anchor)
    return tf.concat(anchors, axis=0)

def data_read(imgs_path: str, anns_path: str, img_size: '(int, int)' = (624, 624)) -> 'tf.data.Dataset':
    anns = [path.join(anns_path, ann_file) for ann_file in os.listdir(anns_path)]
    imgs_folder = imgs_path

    def parse_xml(root):
        x1y1_x2y2_c = []
        for member in root.findall('object'):
            mclass = (0.0 if 'im' in member[0].text else 
                     (1.0 if 'el' in member[0].text else -1.0)) 
            xmin, ymin, xmax, ymax = member.find("bndbox")
            x1y1 = [float(xmin.text) , float(ymin.text) ]
            x2y2 = [float(xmax.text) , float(ymax.text) ]
            x1y1_x2y2_c.append(x1y1 + x2y2 + [mclass])
        return x1y1_x2y2_c 

    def imglab_read():
        for ann in anns:
            root = ET.parse(ann).getroot()

            img_name = root.find('filename').text
            img_name = path.join(imgs_folder, img_name)
            img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB) 
            boxes = parse_xml(root)
            yield tf.constant(img, "float32"), tf.constant(boxes, "float32")

    width, height = img_size
    ds = tf.data.Dataset.from_generator(
            imglab_read,
            output_signature = (tf.TensorSpec((width, height, 3), "float32"),
                                tf.TensorSpec((None, 5), "float32")) 
            )
    return ds

def data_preprocess(ds, new_size=(256, 256)):
    def img_pre(img, boxes):
        img = img/255.0
        width, height = img.shape[0], img.shape[1]
        if (height != new_size[1]) or (width != new_size[0]):
            img = tf.image.resize(img, new_size)
            x1 = boxes[:,0]*(new_size[0]/width)
            y1 = boxes[:,1]*(new_size[1]/height)
            x2 = boxes[:,2]*(new_size[0]/width)  
            y2 = boxes[:,3]*(new_size[1]/height)
            boxes = tf.stack([x1, y1, x2, y2, boxes[:,4]], axis=1)
        return img, boxes

    pre_ds = ds.map(img_pre)
    return pre_ds

def data_encode(ds, featuremap_sizes, aspect_ratios, thresh = 0.5):
    def enc_anchors(img, boxes):
        anchors = []
        for featmap_size in featuremap_sizes:
            for aspect_ratio in aspect_ratios:
                anchor = generate_anchors(featmap_size, img.shape[0:2], aspect_ratio)
                anchor = tf.reshape(anchor, [-1, 4])
                anchors.append(anchor)
        anchors_xywh = tf.concat(anchors, axis=0)
        anchors_corners = convert_to_corners(anchors_xywh)
        IoU_matrix = tf.py_function(get_IoUs, 
                              inp=(anchors_corners, boxes[:, 0:4]),
                              Tout=tf.float32)

        max_IoUs = tf.reduce_max(IoU_matrix, axis=1)
        max_IoUs_ids = tf.argmax(IoU_matrix, axis=1)
        max_IoUs_ids = tf.cast(max_IoUs_ids, tf.int32)
        obj_mask = tf.greater_equal(max_IoUs, thresh)

        boxes_classes = boxes[:, 4]

        class_ids = tf.gather(boxes_classes, max_IoUs_ids)
        class_ids = tf.cast(class_ids, tf.float32)
        class_ids = tf.where(obj_mask, class_ids, -1.0)
        class_ids = tf.expand_dims(class_ids, axis=1)

        boxes_xywh = convert_to_xyhw(boxes)
        gt_boxes_xywh = tf.gather(boxes_xywh[:, 0:4], max_IoUs_ids)
        encoded_anchors = tf.concat(
                [
                    (gt_boxes_xywh[:,0:2] - anchors_xywh[:, 0:2]) / anchors_xywh[:, 2:],
                    tf.math.log(gt_boxes_xywh[:, 2:]/anchors_xywh[:, 2:])
                ],
                axis = 1
                )
        matched_anchors = tf.concat([encoded_anchors, class_ids], axis=-1)
        return img, matched_anchors
    enc_ds = ds.map(enc_anchors)
    return enc_ds

if __name__ == "__main__":
    IMG_PATH = "../data/GermPredDataset/ZeaMays/img"
    ANNS_PATH = "../data/GermPredDataset/ZeaMays/true_ann"
    ds = data_read(IMG_PATH, ANNS_PATH)
    pre_ds = data_preprocess(ds, (256, 256))
    featuremap_sizes = [(64, 64), (32, 32), (8, 8)]
    aspect_ratios = (1,  2/3, 3/2)
    enc_ds = data_encode(pre_ds, featuremap_sizes, aspect_ratios, thresh=0.35)

    import matplotlib.pyplot as plt
    for x, y in enc_ds.take(2):
        anchors = generate_multiple_anchors(featuremap_sizes, (256, 256), aspect_ratios)
        phi = y[:, :4]
        cls = tf.expand_dims(y[:, 4], axis=-1)
        xy = phi[:, 0:2]*anchors[:, 2:] + anchors[:, :2]
        wh = tf.math.exp(phi[:, 2:]) * anchors[:, 2:]
        xywh_cls = tf.concat([xy, wh, cls], axis=1)
        xyxy_cls = convert_to_corners(xywh_cls)
        q = bndboxes_draw(255*x, xyxy_cls[xyxy_cls[:, 4] != -1])
        print(y[y[:, 4]!=0].shape[0])
        plt.imshow(q); plt.show()
