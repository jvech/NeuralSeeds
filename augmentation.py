#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from os import path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copy2, move
from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('img_path', './DatasetE2/JPEGImages', 'path for input images')
flags.DEFINE_string('masks_path', './DatasetE2/SegmentationClass', 'path for label images')
flags.DEFINE_string('augmented_path', None, 'path for augmented dataset')
flags.DEFINE_string('labels','./DatasetE2/labelmap.txt','path for the labels description')
flags.DEFINE_integer('n_images',192,'number of images to generate')

# Data generator:
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=180,
    width_shift_range=0.5,
    height_shift_range=0.5,
    zoom_range = 0.7,
    shear_range = 45,
    vertical_flip = True,
    horizontal_flip = True
    )

def main(argv_):

    # Initialize variables

    ImgDir = FLAGS.img_path.strip('/')
    MasksDir = FLAGS.masks_path.strip('/')
    results_path = FLAGS.augmented_path.strip('/')
    labels_path = FLAGS.labels
    n_images = FLAGS.n_images

    if not FLAGS.augmented_path:
        try:
            os.mkdir('./AugmentedDataset')
        except FileExistsError:
            pass
        results_path = './AugmentedDataset'
    else:
        try:
            os.mkdir(results_path)
            os.mkdir(results_path+'/Images')
            os.mkdir(results_path+'/Masks')
        except FileExistsError:
            pass

    copy2(labels_path,results_path)

    IMG_PATH = path.join(ImgDir, os.listdir(ImgDir)[0])
    image_size = plt.imread(IMG_PATH).shape[:2]

    DATA_PATH = path.abspath(path.join(ImgDir, '..')) 
    seed = np.random.randint(100)
    IMGS_OUT = path.join(results_path, 'Images')
    IMGS_SUBDIR = ImgDir.strip("/").split("/")[-1]
    image_generator = datagen.flow_from_directory(directory = DATA_PATH,
                                                  target_size = image_size,
                                                  save_to_dir = IMGS_OUT,
                                                  classes = [IMGS_SUBDIR],
                                                  class_mode = None,
                                                  save_format = 'jpg',
                                                  seed = seed)

    DATA_PATH = path.abspath(path.join(ImgDir, '..')) 
    MASK_OUT = path.join(results_path, 'Masks')
    MASKS_SUBDIR = MasksDir.strip("/").split("/")[-1]
    masks_generator = datagen.flow_from_directory(directory = DATA_PATH,
                                                  target_size = image_size,
                                                  save_to_dir = MASK_OUT,
                                                  classes = [MASKS_SUBDIR],
                                                  class_mode = None,
                                                  save_format = 'png',
                                                  seed = seed)


    n_iter = int(n_images/32)
    if n_images % 32:
        n_iter += 1

    print(f"Generating {n_iter*32} images")
    for i in range(n_iter):
        image_generator.next()
        masks_generator.next()
        print(i+1,'/',n_iter, 'done')

if __name__=="__main__":
    app.run(main)
