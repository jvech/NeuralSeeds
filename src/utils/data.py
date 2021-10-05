import os
from os import path
import cv2
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET

def get_xyhw(box_corners):
    xmin = box_corners[:, 0] 
    ymin = box_corners[:, 1] 
    xmax = box_corners[:, 2] 
    ymax = box_corners[:, 3] 
    classes = box_corners[:, -1]

    w = xmax - xmin
    h = ymax - ymin

    x = xmin + w/2
    y = ymin + h/2
    return tf.stack([x, y, w, h, classes], axis=1)

def get_corners(box_xyhw):
    x = box_xyhw[:, 0]
    y = box_xyhw[:, 1]
    w = box_xyhw[:, 2]
    h = box_xyhw[:, 3]
    classes = box_xyhw[:, -1]

    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2

    return tf.stack([xmin, ymin, xmax, ymax, classes], axis=1)

def bndboxes_draw(img: "tf.Tensor", boxes: "tf.Tensor") -> "np.ndarray":
    X = img.numpy().astype("uint8").copy()
    Y = boxes.numpy().astype("uint").copy()
    for x1, y1, x2, y2, c in Y:
        pt1, pt2 = (x1, y1), (x2, y2)
        if c == 1: COLOR = (255, 0, 0)
        elif c == 2: COLOR = (255, 255, 0)
        else: COLOR = (0, 0, 0)
        X = cv2.rectangle(X, pt1, pt2, COLOR, 2)
    return X

def data_read(imgs_path: str, anns_path: str, img_size: '(int, int)' = (624, 624)) -> 'tf.data.Dataset':
    anns = [path.join(anns_path, ann_file) for ann_file in os.listdir(anns_path)]
    imgs_folder = imgs_path

    def parse_xml(root):
        x1y1_x2y2_c = []
        for member in root.findall('object'):
            mclass = (1.0 if 'im' in member[0].text else 
                     (2.0 if 'el' in member[0].text else 0)) 
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

def data_preprocess(ds, new_size=(300, 300)):
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

def ds_encode():
    #TODO
    pass

if __name__ == "__main__":
    IMG_PATH = "../data/GermPredDataset/ZeaMays/img"
    ANNS_PATH = "../data/GermPredDataset/ZeaMays/true_ann"
    ds = data_read(IMG_PATH, ANNS_PATH)
    pre_ds = data_preprocess(ds)

    import matplotlib.pyplot as plt
    for x, y in pre_ds.take(5):
        box_xywh = get_xyhw(y)
        corner = get_corners(box_xywh)
        plt.imshow(bndboxes_draw(x, y)); plt.show()
