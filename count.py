#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Dataset import mask2categorical, parse_labelfile
from sys import argv, stderr

USAGE = f"USAGE [OPTIONS ...] [PNG_IMG] [TXT_FILE]\n"
USAGE += "Count the segmented objects of a predicted or true mask image\n\n"
USAGE += "OPTIONS:\n\t"
USAGE += "-v --verbose: show the boxes around the pixels in the image\n\t"

def boxes(stats):
    boxes = []
    for i,stat in enumerate(stats):
        if stat[-1] > 1200 and i!=0:
            boxes.append(stat[:-1])
    return np.array(boxes)

def count_seeds(img, boxes, labels):
    Img = img.copy()
    if Img.dtype != "uint8":
        Img = (255 * Img).astype("uint8")

    count = [0]*(len(labels) - 1)
    for i, box in enumerate(boxes):
        y, x, h, w = box
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        img_box = Img[x1:x2, y1:y2]
        S = [(img_box == labels[key]).sum() 
                for key in labels]
        j = np.argmax(S[1:])
        count[j] += 1
    label_keys = list(labels.keys())[1:]
    return {key: count[i] for i, key in enumerate(label_keys)}

def draw_boxes(img, boxes):
    X = (img.copy()*255).astype("uint8")
    for i, box in enumerate(boxes):
        pt1 = box[:2]
        pt2 = pt1 + box[2:4] 
        pt1 = tuple(pt1.tolist())
        pt2 = tuple(pt2.tolist())
        COLOR = tuple([255, 255, 255])
        X = cv2.rectangle(X, pt1, pt2, COLOR)
    return X

if __name__ == "__main__":
    args = [arg for arg in argv[1:] if not arg.startswith("-")]
    opts = [opt for opt in argv[1:] if opt.startswith("-")]

    if len(args) == 2:
        IMG_PATH = tuple(arg for arg in args if arg[-3:] == "png")[0]
        LABEL_PATH = tuple(arg for arg in args if arg[-3:] == "txt")[0]
        LABELS = parse_labelfile(LABEL_PATH)

        img = plt.imread(IMG_PATH)
        img_int = (img*255).astype("uint8")
        mask = mask2categorical(img, LABELS).numpy()
        count, n_img, stats, centroids = cv2.connectedComponentsWithStats(mask)
        boxes = boxes(stats)
        dimg = draw_boxes(img, boxes)
        lcount = count_seeds(img_int, boxes, LABELS)
        print(f"germinated:\t {lcount['germinated']}")
        print(f"no_germinated:\t {lcount['no_germinated']}")
        if "-v" in opts or "--verbose" in opts:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(img)
            ax[1].imshow(dimg)
            title = f""
            for key in lcount:
                title += f"{key}: {lcount[key]}  "
            fig.suptitle(title)
            plt.show()
    else:
        print(USAGE, file=stderr)

