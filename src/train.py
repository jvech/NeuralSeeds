#!/usr/bin/python3
""" Train
Usage:
    train.py [options] <imgs> <masks> <labelmap.txt>

Options:
    -h --help               Show this message
    -H --history            Plot the history of model's performance
    --model <file>          Save the trained model [default: ./model.h5]
    --batch <int>           Batch size [default: 8]
    --epochs <int>          Number of epochs [default: 40]
    --val_split <float>     Rate of the validation data [default: 0.0]
"""

import os
try: import silence_tensorflow.auto
except ModuleNotFoundError: pass
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from docopt import docopt
from os import path
from utils import data
from utils.model import get_model

from tensorflow.keras.losses import SparseCategoricalCrossentropy

def train(args):
    ## CLI ARGS
    IMG_PATH = args["<imgs>"]
    LABEL_PATH = args["<masks>"]
    IMG_PATHS = np.sort([path.join(IMG_PATH, img) for img in os.listdir(IMG_PATH)])
    LABEL_PATHS = np.sort([path.join(LABEL_PATH, img) for img in os.listdir(LABEL_PATH)])

    MODEL_PATH = args["--model"]

    LABEL_ID = data.parse_labelfile(args["<labelmap.txt>"])
    BATCH_SIZE = int(args["--batch"])
    EPOCHS = int(args["--epochs"])
    SPLIT_RATE = float(args["--val_split"])
    SHAPE = (224, 224)

    ds = data.read_data(IMG_PATHS, LABEL_PATHS)
    ds_train = data.preprocess_ds(ds, LABEL_ID, SHAPE)

    ds_train = ds_train.shuffle(buffer_size=50)
    if SPLIT_RATE > 0.0:
        ds_train, ds_val = data.train_val_split(ds_train, SPLIT_RATE)
        ds_val = ds_val.map(lambda x, y: (x[tf.newaxis], y[tf.newaxis]))
    else:
        ds_val = None
    ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)

    model = get_model(output_channels=3, size=list(SHAPE) + [3])
    model.compile(
            optimizer = "adam",
            metrics = ["accuracy"],
            loss = SparseCategoricalCrossentropy()
            )

    DS_LEN = len(list(ds_train.as_numpy_iterator()))
    model_hist = model.fit(ds_train, 
                           validation_data = ds_val,
                           epochs = EPOCHS,
                           steps_per_epoch = DS_LEN
                           )

    model.save(MODEL_PATH)
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    args = docopt(__doc__)
    train(args)
