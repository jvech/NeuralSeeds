#!/usr/bin/python3
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR) ## Disable tf Warnings
from absl import app, flags, logging
from absl.flags import FLAGS
from model import get_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

flags.DEFINE_string("image_path", "./seeds_data/JPEGImages/004.jpg", "input image path")
flags.DEFINE_string("mask_path", None, "path save the predicted mask")
flags.DEFINE_string("weights", "./weights/cp-0010.ckpt", "weights parameters path")
flags.DEFINE_bool("show_results", True, "show prediction result")

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def main(_argv):
    img_path = FLAGS.image_path
    out_path = FLAGS.mask_path
    weights_path = FLAGS.weights
    show_results = FLAGS.show_results

    model = get_model(output_channels=1, size=224)
    model.load_weights(weights_path)
    X = plt.imread(img_path)/255
    X = tf.convert_to_tensor(X)
    X = tf.expand_dims(X, 0)
    X = tf.image.resize(X, (224, 224))
    Y = model.predict(X)
    if show_results:
        display([X[0], Y[0]])

    if out_path != None:
        plt.imsave(out_path, Y[0])
        logging.info(f"{out_path} saved")

if __name__ == "__main__":
    app.run(main)
