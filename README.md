# U-Net network to segment seeds

A neural network designed to segmentate 3 different classes, by default it was
thought to segmentate germinated seeds, no germinated seeds and the background

The principal scripts are:
* train.py
* make\_TFrecords.py
* predict.py

## Train model
To train the model you must first create two tfrecords file using the script
make_TFrecords.py next you can use the train.py file to train your model

## Predict an image
Use the predict.py script to predict an image, we recommend that the predicted
mask be saved in png format instead of jpg

## Check the performance
To see how good was the prediction use measure.py, it will print the dice and
jaccard indexes of the prediction respect to the mask,

## TO DO:

Add the count part to the project
