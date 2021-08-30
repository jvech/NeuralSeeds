import tensorflow as tf 
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Input,
                                     Conv2DTranspose, 
                                     Conv2D,
                                     MaxPool2D,
                                     Dropout, 
                                     ReLU, 
                                     BatchNormalization,
                                     Concatenate)
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.ops.gen_array_ops import size


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

def get_model(out_channels=3, in_size=[128, 128, 3], name="U-Net"):
    x = inputs = Input(shape=in_size)

    skips = get_encoder(input_shape=list(x.shape[1:]))(x)

    x = get_decoder(skips)

    last = Conv2DTranspose(
                    filters=out_channels, 
                    kernel_size=3, 
                    strides=2,
                    padding='same',
                    activation=tf.keras.activations.softmax)  #64x64 -> 128x128
    x = last(x)
    return Model(inputs=inputs, outputs=x, name=name)

def UNET(out_channels:int = 3, in_size:tuple = (256, 256, 3)):
    def ENC_CNN(in_layer, filters):
        conv = Conv2D(filters, 3, activation = "relu", padding = "same")(in_layer)
        conv = Conv2D(filters, 3, activation = "relu", padding = "same")(conv)
        pool = MaxPool2D(pool_size = (2, 2))(conv)
        return pool, conv

    def DEC_CNN(in_layer, skip_layer, filters):
        up = UpSampling2D(size = (2, 2))(in_layer)
        up = Conv2DTranspose(filters, 2, padding = "same")(up)
        conc = Concatenate()([skip_layer, up])
        conv = Conv2D(filters, 3, activation = "relu", padding = "same")(conc)
        conv = Conv2D(filters, 3, activation = "relu", padding = "same")(conv)
        return conv

    in_layer = Input(shape=in_size)

    # ENCODER
    block1, skip1 = ENC_CNN(in_layer, 64)
    block2, skip2 = ENC_CNN(block1, 128)
    block3, skip3 = ENC_CNN(block2, 256)
    block4, skip4 = ENC_CNN(block3, 512)

    block5 = Conv2D(1024, 3, activation = "relu", padding = "same")(block4)
    block5 = Conv2D(1024, 3, activation = "relu", padding = "same")(block5)

    # DECODER
    block6 = DEC_CNN(block5, skip4, 512)
    block7 = DEC_CNN(block6, skip3, 256)
    block8 = DEC_CNN(block7, skip2, 128)
    block9 = DEC_CNN(block8, skip1, 64)

    out_layer = Conv2D(out_channels, 1, activation="softmax")(block9)

    return Model(inputs = in_layer, outputs=out_layer)


if __name__=="__main__":
    #app._run_init(['unet'],app.parse_flags_with_usage)
    model = get_model(out_channels=3, size=(224, 224))
    model.summary()
    tf.keras.utils.plot_model(model, to_file='data/model.png', rankdir="TB",
                              show_shapes=False, show_layer_names=True, expand_nested=True)
