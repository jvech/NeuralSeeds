import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

IMAGE_FEATURE_MAP = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
        }

filename = './tfrecords/train-data.tfrecord'

def parse_dataset(tfrecord, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP) 
    X_train = tf.image.decode_jpeg(x['image'], channels=3)
    Y_train = tf.image.decode_png(x['mask'])

    X_train = tf.image.resize(X_train, (size, size))
    Y_train = tf.image.resize(Y_train, (size, size))
    return X_train/255, Y_train

def load_tfrecord_dataset(dataset_path, size):
    raw_dataset = tf.data.TFRecordDataset([dataset_path])
    return raw_dataset.map(lambda x: parse_dataset(x, size))

def transform_images(X, Y):
    Y = tf.one_hot(Y, depth=len(np.unique(Y).shape))
    return X, Y

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

def mask2categorical(Mask: tf.Tensor, labels: dict) -> tf.Tensor:
    """Pass a certain rgb mask (3-channels) to an image of ordinal classes"""
    assert type(labels) == dict, "labels variable should be a dictionary"

    X = Mask

    if X.dtype == "float32":
        X = tf.cast(X*255, dtype="uint8")

    Y = tf.zeros(X.shape[0:2] , dtype="float32")
    for i, key in enumerate(labels):
        Y = tf.where(np.all(X == labels[key], axis=-1), i, Y)
    Y = tf.cast(Y, dtype="uint8")
    return Y
if __name__=="__main__":
    size = 512
    train_dataset = load_tfrecord_dataset(filename, size)

    for X_train, Y_train in train_dataset.take(2):
        plt.subplot(121)
        plt.imshow(X_train)
        plt.subplot(122)
        plt.imshow(Y_train)
        plt.show()


