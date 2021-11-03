#!/usr/bin/python3
""" Train
Usage:
    train.py [options] <imgs> <annotations> 

Options:
    -h --help               Show this message
    -H --history            Show the history of model metrics
    --model <file>          Save the trained model [default: ./model.h5]
    --batch <int>           Batch size [default: 8]
    --epochs <int>          Number of epochs [default: 40]
    --val_split <float>     Rate of the validation data [default: 0.0]
    --backbone <name>       Select the detector backbone [default: mobilenetv2]
"""

import os
try: import silence_tensorflow.auto
except ModuleNotFoundError: pass
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

from utils import data
from utils.models import get_model
from utils.losses import BoxClassLoss

def train(args):
    IMG_PATH  = args["<imgs>"]
    ANN_PATH  = args["<annotations>"]
    BATCH_SIZE = int(args["--batch"])
    EPOCHS = int(args["--epochs"])
    VAL_SPLIT = float(args["--val_split"])
    BACKBONE = args["--backbone"]
    IMG_SIZE = (256, 256) 

    ds = data.data_read(IMG_PATH, ANN_PATH)
    pre_ds = data.data_preprocess(ds, IMG_SIZE)

    grid_sizes = [(8, 8), (16, 16), (32, 32)]
    aspect_ratios = (1, 2/3, 3/2)
    enc_ds = data.data_encode(pre_ds,
                              grid_sizes,
                              aspect_ratios,
                              thresh=0.3)

    enc_ds = enc_ds.shuffle(1024, seed=42)

    for DS_SIZE, _ in enumerate(enc_ds): pass
    else: DS_SIZE+=1

    if VAL_SPLIT > 0.01:
        VAL_SIZE = int(VAL_SPLIT * DS_SIZE)
        TRAIN_SIZE = DS_SIZE - VAL_SIZE
        val_ds = enc_ds.take(VAL_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE).cache()
        train_ds = enc_ds.skip(VAL_SIZE).batch(1).prefetch(tf.data.experimental.AUTOTUNE).cache()
    else:
        TRAIN_SIZE = DS_SIZE
        train_ds = enc_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE).cache()
        val_ds = None

    model = get_model(IMG_SIZE + (3,), BACKBONE, num_classes=2, trainable=False)
    loss_fn = BoxClassLoss(2)
    model.compile(
            loss=loss_fn,
            optimizer="Adam"
            )

    model.fit(
            train_ds.repeat(),
            validation_data=val_ds,
            steps_per_epoch=TRAIN_SIZE//BATCH_SIZE, 
            epochs=EPOCHS
            )
    return model

if __name__ == "__main__":
    args = docopt(__doc__)
    model = train(args)
    model.save(args["--model"])
    pass
