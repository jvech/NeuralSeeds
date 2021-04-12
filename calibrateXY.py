#!/usr/bin/python3
import cv2
import sys
from sys import argv

USAGE = f"USAGE [IMG]\n"
USAGE += "Create a text file with the X Y spatial dimensions per each pixel, \n"
USAGE += "the image file should have known distances references in X and Y axis\n\t"

# Variables
ix = -1
iy = -1
drawing = False
X_AXIS_FINISHED = Y_AXIS_FINISHED = False

# Constants
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
WINNAME = "Spatial XY Dimensions Coordinates"

def draw_Xaxis(event, x, y, flags, param):
    global ix, iy, drawing, img, X_AXIS_FINISHED, pdx

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y            
        cv2.circle(img, center=(x, y),
                    radius=10, 
                    color=GREEN_COLOR,
                    thickness=-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, pt1 =(ix, iy),
                          pt2 =(x, iy),
                          color =RED_COLOR,
                          thickness =4)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, pt1 =(ix, iy),
                      pt2 =(x, iy),
                      color =RED_COLOR,
                      thickness =4)
        X_AXIS_FINISHED = True
        pdx = abs(ix - x)

def draw_Yaxis(event, x, y, flags, param):
    global ix, iy, drawing, img, pdy, Y_AXIS_FINISHED

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y            
        cv2.circle(img, center=(x, y),
                    radius=10, 
                    color=GREEN_COLOR,
                    thickness=-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, pt1 =(ix, iy),
                          pt2 =(ix, y),
                          color =RED_COLOR,
                          thickness =4)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, pt1 =(ix, iy),
                      pt2 =(ix, y),
                      color =RED_COLOR,
                      thickness =4)
        pdy = abs(iy - y)
        Y_AXIS_FINISHED = True

def get_inputdistance(Log):
    while True:
        distance = input(Log)
        try:
            return int(distance)
        except ValueError:
            print(f"{distance} is an invalid input try again", file=sys.stderr)

if __name__ == "__main__":
    if len(argv[1:]) == 2:
        INPUT_IMAGE = argv[1]
        OUTPUT_FILE = argv[2]
        assert INPUT_IMAGE[-3:] == "jpg" or INPUT_IMAGE[-3:] == "png", "Only png or jpg images allowed"
        X = cv2.imread(INPUT_IMAGE)
        while not X_AXIS_FINISHED:
            cv2.namedWindow(WINNAME, cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback(WINNAME, draw_Xaxis)
            img = X.copy()
            while True:
                cv2.imshow(WINNAME, img)

                KEY_PRESS = cv2.waitKey(10)
                if KEY_PRESS == 27:
                    sys.exit()
                if X_AXIS_FINISHED:
                    break
            cv2.destroyAllWindows()
            finished = input("X axis Finished (Y/n): ")
            X_AXIS_FINISHED = (True if finished.lower() == "y" else False)
        dx = get_inputdistance("Put the X axis distance (cm): ") * 10 / pdx

        while not Y_AXIS_FINISHED:
            cv2.namedWindow(WINNAME, cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback(WINNAME, draw_Yaxis)
            img = X.copy()
            while True:
                cv2.imshow(WINNAME, img)

                KEY_PRESS = cv2.waitKey(10)
                if KEY_PRESS == 27:
                    sys.exit()
                if Y_AXIS_FINISHED:
                    break
            cv2.destroyAllWindows()
            finished = input("Y axis Finished (Y/n): ")
            Y_AXIS_FINISHED = (True if finished.lower() == "y" else False)
        dy = get_inputdistance("Put the Y axis distance (cm): ") * 10 / pdy

        file = open("weights/config.csv", "w")
        file.write(f"dx(mm),dy(mm)\n{dx},{dy}\n")
        file.close()
        print("X Y dimensions saved in weights/config.csv")
    else:
        print(USAGE, file=sys.stderr)
