#!/usr/bin/python3
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

from numpy import random
from os import path
from absl import flags, app, logging
from absl.flags import FLAGS
from Dataset import parse_labelfile, mask2categorical

flags.DEFINE_string('img_path', './seeds_data/JPEGImages/', 'path for input images')
flags.DEFINE_string('mask_path', './seeds_data/SegmentationClass/', 'path for label images')
flags.DEFINE_string('tfrecord_path', './tfrecords/', 'path for final tf_record')
flags.DEFINE_float('val_size', 0.2, 'validation data size, it must be a number between: (0.0-1.0)')
flags.DEFINE_string('labels', 'labelmap.txt', 'path to the labels description')

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def train_test_split(imgs_path, masks_path, val_size):
    """Return 4 lists of paths in the following order:
    - Train images
    - Train masks
    - Validation images
    - Validation masks
    """
    train_img_path = []
    train_mask_path = []
    val_img_path = []
    val_mask_path = []

    indexes = []
    n_imgs = len(imgs_path)
    n_val_imgs = int(val_size*n_imgs)
    while len(indexes) < n_imgs:
        x = random.randint(n_imgs)
        if x not in indexes:
            indexes.append(x)

    for i in indexes[0:n_val_imgs]:
        val_img_path.append(imgs_path[i])
        val_mask_path.append(masks_path[i])

    for i in indexes[n_val_imgs:]:
        train_img_path.append(imgs_path[i])
        train_mask_path.append(masks_path[i])

    return train_img_path, train_mask_path, val_img_path, val_mask_path

def create_example(img, mask, labels):
    """Creates a tensorflow Example from an image with its mask.
    Parameters
    -----------
    img : str
        path to the image
    mask : str
        path to the mask
    labels : dict
        dict with the corresponding rgb mask values of the labels
    Returns
    --------
    tf.train.Example   
    """
    encoded_img = tf.io.read_file(img)
    encoded_mask = tf.io.read_file(mask)
    ## mask preprocessing ##
    decoded_mask = tf.io.decode_image(encoded_mask)
    mask = mask2categorical(decoded_mask, labels)
    mask = tf.expand_dims(mask, axis=-1)

    encoded_mask = tf.io.encode_png(mask) # Re-encoding the mask

    example = tf.train.Example(
                features=tf.train.Features(feature={
                    'image': bytes_list_feature(encoded_img.numpy()),
                    'mask': bytes_list_feature(encoded_mask.numpy())
                    }))
    return example


def main(_argv):
    logging.info("Initializing Variables")
    PATH_IMG = FLAGS.img_path
    PATH_MASK = FLAGS.mask_path
    VAL_TFRECORD = path.join(FLAGS.tfrecord_path, "val-data.tfrecord")
    TRAIN_TFRECORD = path.join(FLAGS.tfrecord_path, "train-data.tfrecord")
    LABELS = parse_labelfile(FLAGS.labels)
    val_size = FLAGS.val_size

    # Create 2 lists containing the paths of the images and the masks
    img_path = [path.join(PATH_IMG, imgs) for imgs in np.sort(os.listdir(PATH_IMG))]
    mask_path = [path.join(PATH_MASK, imgs) for imgs in np.sort(os.listdir(PATH_MASK))]

    logging.info("Spliting Data Files")
    train_img, train_mask, val_img, val_mask = train_test_split(img_path, mask_path, val_size)

    logging.info("Writing tfrecords files")
    # Create and fill the train-data.tfrecord with the examples 
    tf_record_train = tf.io.TFRecordWriter(TRAIN_TFRECORD)
    for img, mask in zip(train_img, train_mask):
        example = create_example(img, mask, LABELS)
        tf_record_train.write(example.SerializeToString())
    tf_record_train.close()

    #Create and fill the val-data.tfrecord with the examples
    tf_record_val = tf.io.TFRecordWriter(VAL_TFRECORD)
    for img, mask in zip(val_img, val_mask):
        example = create_example(img, mask, LABELS)
        tf_record_val.write(example.SerializeToString())
    tf_record_val.close()

if __name__ == "__main__":
    app.run(main)
