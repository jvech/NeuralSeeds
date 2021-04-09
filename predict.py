#!/usr/bin/python3
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) ## Disable tf Warnings
from absl import app, flags, logging
from absl.flags import FLAGS
from model import get_model
import tensorflow as tf
import matplotlib.pyplot as plt
from Dataset import parse_labelfile
import numpy as np
import cv2

flags.DEFINE_string("image_path", "./seeds_data/JPEGImages/004.jpg", "input image path")
flags.DEFINE_string("mask_path", None, "path save the predicted mask (recomended file extension: png)")
flags.DEFINE_string("weights", "./weights/cp-0010.ckpt", "weights parameters path")
flags.DEFINE_string("labels", "", "path to the annotation file")
flags.DEFINE_bool("show_results", True, "show prediction result")

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def categorical2mask(X, labels):
    Y = np.zeros(X.shape[0:2] + [3], dtype="uint8")
    for i, key in enumerate(labels):
        Y[...,0] = np.where(X==i, labels[key][0], Y[...,0])
        Y[...,1] = np.where(X==i, labels[key][1], Y[...,1])
        Y[...,2] = np.where(X==i, labels[key][2], Y[...,2])
    return Y

def main(_argv):
    img_path = FLAGS.image_path
    out_path = FLAGS.mask_path
    weights_path = FLAGS.weights
    show_results = FLAGS.show_results
    img_size = 224
    classes = 3
    LABELS_PATH = FLAGS.labels

    labels = parse_labelfile(LABELS_PATH)

    img = plt.imread(img_path)/255
    X = tf.convert_to_tensor(img)
    X = tf.image.resize(X, (img_size, img_size))
    X = tf.expand_dims(X, 0)

    model = get_model(output_channels=classes, size=224)
    model.load_weights(weights_path)

    Y = model.predict(X)
    Y = tf.argmax(Y, axis=-1)
    Y = categorical2mask(Y[0], labels)
    Y = cv2.resize(Y, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if show_results:
        display([img, Y])

    if out_path != None:
        Y = cv2.cvtColor(Y, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_path, Y)
        logging.info(f"{out_path} saved")

if __name__ == "__main__":
    app.run(main)
