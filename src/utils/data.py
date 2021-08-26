import os
from os import path
import cv2
import tensorflow as tf
import numpy as np

def mask2categorical(mask: "tf.Tensor", labels: dict) -> "tf.Tensor":
    """Pass a certain rgb mask (3-channels) to an image of ordinal classes"""

    assert type(labels) == dict, "labels variable should be a dictionary"
    X = mask

    if X.dtype == "float32":
        X = tf.cast(X * 255, dtype="uint8")

    Y = tf.zeros(X.shape[0:2] , dtype="int32")
    for i, key in enumerate(labels):
        Y = tf.where(tf.reduce_all(X == tf.constant(labels[key], "uint8"), axis=-1), i, Y)
    Y = tf.cast(Y, dtype=tf.uint8)
    return tf.expand_dims(Y, axis=-1)

def categorical2mask(categorical: "tf.Tensor", labels: dict) -> "tf.Tensor":
    X = categorical
    Y = np.zeros(X.shape[0:2] + [3], dtype="uint8")
    for i, key in enumerate(labels):
        Y[...,0] = np.where(X==i, labels[key][0], Y[...,0])
        Y[...,1] = np.where(X==i, labels[key][1], Y[...,1])
        Y[...,2] = np.where(X==i, labels[key][2], Y[...,2])
    return Y

def parse_labelfile(path):
    """Return a dict with the corresponding rgb mask values of the labels
        Example:
        >>> labels = parse_labelfile("file/path")
        >>> print(labels) 
        >>> {"label1": (r1, g1, b1), "label2": (r2, g2, b2)} 
    """
    with open(path, "r") as FILE:
        lines = FILE.readlines()

    labels = {x.split(":")[0]: x.split(":")[1] for x in lines[1:]}

    for key in labels:
        labels[key] = np.array(labels[key].split(",")).astype("uint8")

    return labels

## Dataset Functions
def read_data(img_paths, label_paths):
    def imgs_read():
        for X, Y in zip(img_paths, label_paths):
            yield (tf.constant(cv2.imread(X)[..., ::-1], tf.uint8),
                   tf.constant(cv2.imread(Y)[..., ::-1], tf.uint8))
    IMG_DIMS = cv2.imread(img_paths[0]).shape
    ds = tf.data.Dataset.from_generator(
            imgs_read,
            output_signature= (tf.TensorSpec(IMG_DIMS, tf.uint8),
                               tf.TensorSpec(IMG_DIMS, tf.uint8))
            )
    return ds

def preprocess_ds(ds: "tf.data.Dataset", labelmap: dict, shape: tuple) ->  "tf.data.Dataset":
    def normalize(img, label):
        return tf.cast(img, tf.float32) / 255., label

    def set_labels(img, label):
        label_out = mask2categorical(label, labelmap)
        return img, label_out

    def resize(img, label):
        X = tf.image.resize(img, shape)
        Y = tf.image.resize(label, shape, method='nearest' )
        return X, Y

    pre_ds = ds.map(normalize)
    pre_ds = pre_ds.map(resize)
    pre_ds = pre_ds.map(set_labels)
    return pre_ds

def train_val_split(ds: "tf.data.Dataset", val_size: float = 0.2):
    len_ds = len(list(ds.as_numpy_iterator()))
    train_size = int((1 - val_size) * len_ds)

    ds_train = ds.take(train_size)
    ds_val = ds.skip(train_size)
    return ds_train, ds_val

if __name__ == "__main__":
    IMG_PATH = "../data/DatasetE2/JPEGImages"    
    LABEL_PATH = "../data/DatasetE2/SegmentationClass"
    LABELMAP = parse_labelfile("../data/DatasetE2/labelmap.txt")
    IMG_PATHS = np.sort([path.join(IMG_PATH, img) for img in os.listdir(IMG_PATH)])
    LABEL_PATHS = np.sort([path.join(LABEL_PATH, img) for img in os.listdir(LABEL_PATH)])
    ds = read_data(IMG_PATHS, LABEL_PATHS)
    pre_ds = preprocess_ds(ds, LABELMAP, (512, 512))
    pass
