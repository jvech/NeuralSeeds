"""
Implementation of U-Net with backbone MobileNetV2

by: Juan Carlos Aguirre
"""
import tensorflow as tf 
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Input,
                                     Conv2DTranspose, 
                                     Dropout, 
                                     ReLU, 
                                     BatchNormalization,
                                     Concatenate)


def upsample(filters,size,strides=2,padding="same",batchnorm=False,dropout=False):

    layer = Sequential()
    layer.add(
        Conv2DTranspose(filters,size,strides,padding,use_bias = False))

    if batchnorm:
        layer.add(BatchNormalization())

    if dropout:
        layer.add(Dropout(0.5))

    layer.add(ReLU())

    return layer

def get_encoder(input_shape=[None,None,3],name="encoder"): 
    x_in = Input(shape=input_shape)
    base_model = MobileNetV2(input_tensor=x_in, include_top=False)
    layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    encoder  = Model(inputs=x_in, outputs=layers,name=name)
    encoder.trainable = False

    return encoder 

def get_decoder(skips):
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(256, 3),  # 8x8 -> 16x16
        upsample(128, 3),  # 16x16 -> 32x32
        upsample(64, 3),   # 32x32 -> 64x64
    ]
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up,skip in zip(up_stack,skips):
        x = up(x)
        x = Concatenate()([x,skip])
    return x

def get_model(output_channels=3, size=[128, 128, 3], name="U-Net"):
    x = inputs = Input(shape=size)

    skips = get_encoder(input_shape=list(x.shape[1:]))(x)

    x = get_decoder(skips)

    last = Conv2DTranspose(
                    filters=output_channels, 
                    kernel_size=3, 
                    strides=2,
                    padding='same',
                    activation=tf.keras.activations.softmax)  #64x64 -> 128x128
    x = last(x)
    return Model(inputs=inputs, outputs=x, name=name)

if __name__=="__main__":
    #app._run_init(['unet'],app.parse_flags_with_usage)
    model = get_model(output_channels=3, size=224)
    model.summary()
    tf.keras.utils.plot_model(model, to_file='data/model.png', rankdir="TB",
                              show_shapes=False, show_layer_names=True, expand_nested=True)
