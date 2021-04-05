#!/usr/bin/python3
"""
Script That Calculates the Dice and Jaccard Index of two masks images
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from Dataset import parse_labelfile, mask2categorical


def jaccard_index(A: np.array, B: np.array, labels: dict) -> list:
    """IoU(A, B) = (A & B) / (A U B)"""
    assert A.shape == B.shape

    label = [i for i, key in enumerate(labels)]
    IoU = []

    for x in label:
        Inter = np.sum((A == x) & (B == x))
        Union = np.sum((A == x) | (B == x))
        if np.sum(A==x) == 0:
            IoU.append(-1)
        else:
            IoU.append(Inter/Union)
    return IoU

def dice_index(A: np.array, B: np.array, labels: dict) -> list:
    """Dice(A, B) = 2*(A & B) / (|A| + |B|)"""

    label = [i for i, key in enumerate(labels)]
    dice = []

    for x in label:
        Inter = np.sum((A == x) & (B == x))
        Den = (np.sum(A == x) + np.sum(B == x))
        if np.sum(A==x) == 0:
            dice.append(-1)
        else:
            dice.append(2 * Inter / Den)
    return dice

if __name__ == "__main__":
    if(len(sys.argv) == 4):
        LABEL_PATH = sys.argv[-1]
        TRUE_MASK = sys.argv[1]
        PRED_MASK = sys.argv[2]

        labels = parse_labelfile(LABEL_PATH)
        Y_pred = plt.imread(PRED_MASK)
        Y_true = plt.imread(TRUE_MASK)
        Y_pred = mask2categorical(Y_pred, labels)
        Y_true = mask2categorical(Y_true, labels)

        IoU = tuple(jaccard_index(Y_true, Y_pred, labels))
        dice = tuple(dice_index(Y_true, Y_pred, labels))
        IoU_dice = IoU + dice
        print((2*len(labels)-1)*"%1.4f,"%IoU_dice[:-1] + "%1.4f"%IoU_dice[-1])
    else:
        print("Usage: ./measure.py [TRUEMASK] [PREDMASK] [LABELS]", file=sys.stderr)
        print("Show how good was the prediction compared with its true mask", file=sys.stderr)
