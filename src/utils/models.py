import tensorflow as tf
from keras import Model
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Add, Concatenate, Reshape, UpSampling2D, Layer


def get_backbone(input_shape, backbone_name="MobileNetV2", trainable=True, **kwargs):
    name = backbone_name.lower()
    if name == "mobilenetv2":
        model = MobileNetV2(input_shape=input_shape,
                            include_top=False, **kwargs)

        layer_names = ["block_6_expand_relu", "block_13_expand_relu", "out_relu"]
        outs = [model.get_layer(lyr_name).output for lyr_name in layer_names]
    elif name == "vgg16":
        model = VGG16(input_shape=input_shape,
                      include_top=False, **kwargs)
        layer_names = ["block3_pool", "block4_pool", "block5_pool"]
        outs = [model.get_layer(lyr_name).output for lyr_name in layer_names]

    else: raise ValueError("unknown backbone")

    model.trainable = trainable
    return Model(inputs=model.inputs, outputs = outs)

def get_bottleneck(backbone: "keras.Model") -> "keras.Model":
        filters = 128
        out_c1, out_c2, out_c3 = backbone.outputs

        c1_out = Conv2D(filters, 1, 1, "same")(out_c1) #32x32
        c2_out = Conv2D(filters, 1, 1, "same")(out_c2) #16x16
        c3_out = Conv2D(filters, 1, 1, "same")(out_c3) #8x8

        c3_out_upx2 = UpSampling2D(2)(c3_out) #16x16

        c2_c3_sum = Add()([c2_out, c3_out_upx2]) #16x16

        c2_out_upx2 = UpSampling2D(2)(c2_c3_sum) #32x32

        p1_out = Add()([c1_out, c2_out_upx2]) #32x32

        p2_out = Conv2D(filters, 3, 2, "same")(p1_out) #16x16
        p3_out =   Conv2D(filters, 3, 2, "same")(p2_out) #8x8

        outs = [p1_out, p2_out, p3_out]
        return Model(inputs=backbone.inputs, outputs=outs)

def get_head(num_classes, backbone):
    out_layers = []
    for backbone_lyr in backbone.outputs:
        neck_lyrs = []
        for _ in range(3):
            class_lyr = Conv2D(filters=num_classes,
                                kernel_size=1,
                                padding="same",
                                )

            detection_lyr = Conv2D(filters=4,
                                   kernel_size=1,
                                   padding="same")

            neck_lyr = Concatenate(axis=-1)([detection_lyr(backbone_lyr), 
                                             class_lyr(backbone_lyr)])

            neck_lyr = Reshape([-1, 4 + num_classes])(neck_lyr)
            neck_lyrs.append(neck_lyr)
        out_lyr = Concatenate(axis=1)(neck_lyrs)
        out_layers.append(out_lyr)
    out = Concatenate(axis=1)(out_layers)
    return Model(inputs=backbone.inputs, outputs=out)


def get_model(input_shape: tuple, backbone_name: str, num_classes: int, **kwargs) -> "keras.Model":
    tf.keras.backend.clear_session()
    backbone = get_backbone(input_shape, backbone_name, **kwargs)
    pyramid = get_bottleneck(backbone)
    head = get_head(num_classes, pyramid)
    return head


if __name__ == "__main__":
    model = get_model(input_shape = (256, 256, 3),
                      backbone_name = "mobilenetv2",
                      num_classes = 2,
                      trainable=False)
    model.summary()
    plot_model(model, expand_nested="LR")
