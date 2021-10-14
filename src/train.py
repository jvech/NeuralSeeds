#!/usr/bin/python3
""" Train
Usage:
    train.py [options] <imgs> <annotations>

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

from utils import data

def train(args):
    IMG_PATH  = args["<imgs>"]
    ANN_PATH  = args["<annotations>"]
    BATCH_SIZE = int(args["--batch"])
    EPOCHS = int(args["--epochs"])

    ds = data.data_read(IMG_PATH, ANN_PATH)
    pre_ds = data.data_preprocess(ds)

    #TODO
    pass

if __name__ == "__main__":
    args = docopt(__doc__)
    model = train(args)
    model.summary()
    pass
