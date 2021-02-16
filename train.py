#!/usr/bin/python3
from absl import app, flags, logging
from absl.flags import FLAGS
import Dataset 
import tensorflow as tf 
from model import get_model

import matplotlib.pyplot as plt

flags.DEFINE_string('train_Dataset','./tfrecords/train-data.tfrecord','path to train Dataset')
flags.DEFINE_string('val_Dataset','./tfrecords/val-data.tfrecord','path to validation Dataset')
flags.DEFINE_string('weights', './weights/', 'path of the model\'s weights')
flags.DEFINE_integer('buffer_size', 100, 'buffer')
flags.DEFINE_integer('batch_size', 5, 'batch size')
flags.DEFINE_integer('epochs', 10, 'Epochs')
flags.DEFINE_integer('save_freq', 5, 'frequency of epochs to save')


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

    #Initialize Variables
    checkpoint_path = FLAGS.weights+'cp-{epoch:04d}.ckpt'
    image_size = 224 

    ## Loading Dataset
    train_Dataset = Dataset.load_tfrecord_dataset(
                        FLAGS.train_Dataset, image_size)
    train_Dataset = train_Dataset.map(lambda X, Y: Dataset.transform_images(X, Y))
    train_Dataset = train_Dataset.shuffle(buffer_size=FLAGS.buffer_size)
    train_Dataset = train_Dataset.batch(FLAGS.batch_size, drop_remainder=True)
    train_Dataset = train_Dataset.prefetch(
                        buffer_size=tf.data.experimental.AUTOTUNE)

    val_Dataset = Dataset.load_tfrecord_dataset(FLAGS.val_Dataset, image_size)
    val_Dataset = val_Dataset.map(lambda X, Y: Dataset.transform_images(X, Y))
    val_Dataset = val_Dataset.batch(FLAGS.batch_size, drop_remainder=True)

    ## Loading Model
    n_batches_per_epoch = len(list(train_Dataset.as_numpy_iterator()))
    save_freq = int(n_batches_per_epoch * FLAGS.save_freq)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_path,
                                verbose=1,
                                save_weights_only=True,
                                save_freq=save_freq
                                )

    model = get_model(output_channels=1,size=None)
    model.save_weights(checkpoint_path.format(epoch=0))
    model.compile(optimizer='adam', 
                  metrics=['accuracy'],
                  loss = tf.keras.losses.BinaryCrossentropy())

    ## Training Model
    model_history = model.fit(
                        train_Dataset, 
                        epochs=FLAGS.epochs,
                        validation_data=val_Dataset,
                        batch_size=FLAGS.batch_size,
                        callbacks=[cp_callback]
                        )

    for image, mask in val_Dataset.take(-1):
        predict = model.predict(image)
        display([image[0],mask[0],predict[0]])

if __name__=="__main__":
    app.run(main)

