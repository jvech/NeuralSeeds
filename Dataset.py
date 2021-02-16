import tensorflow as tf
import matplotlib.pyplot as plt
IMAGE_FEATURE_MAP = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
        }

filename = './tfrecords/train-data.tfrecord'

def parse_dataset(tfrecord, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP) 
    X_train = tf.image.decode_jpeg(x['image'], channels=3)
    Y_train = tf.image.decode_png(x['mask'], channels=3)

    #X_train = tf.image.resize(X_train, (size, size))
    #Y_train = tf.image.resize(Y_train, (size, size))
    return X_train/255, Y_train/255

def load_tfrecord_dataset(dataset_path, size):
    raw_dataset = tf.data.TFRecordDataset([dataset_path])
    return raw_dataset.map(lambda x: parse_dataset(x, size))

def transform_images(X, Y):
    Y = tf.image.rgb_to_grayscale(Y)
    return X, Y
    

if __name__=="__main__":
    size = 512
    train_dataset = load_tfrecord_dataset(filename, size)

    for X_train, Y_train in train_dataset.take(2):
        plt.subplot(121)
        plt.imshow(X_train)
        plt.subplot(122)
        plt.imshow(Y_train)
        plt.show()
