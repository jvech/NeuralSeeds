import tensorflow as tf
from keras import Model
from tensorflow.keras.layers import Conv2D, Input, Concatenate, Reshape
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, VGG16
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

def get_backbone(input_shape, backbone_name="MobileNetV2"): 
    name = backbone_name.lower()
    if name == "mobilenetv2":
        model = MobileNetV2(input_shape=input_shape,
                            include_top=False)

        layer_names = ["block_6_expand_relu", "block_13_expand_relu", "out_relu"]
        outs = [model.get_layer(lyr_name).output for lyr_name in layer_names]
    elif name == "vgg16":
        model = VGG16(input_shape=input_shape,
                            include_top=False)
        layer_names = ["block3_pool", "block4_pool", "block5_pool"]
        outs = [model.get_layer(lyr_name).output for lyr_name in layer_names]

    else: raise ValueError("unknown backbone")

    return Model(inputs=model.inputs, outputs = outs)

def get_head(num_classes, backbone):
    out_layers = []
    for backbone_lyr in backbone.outputs:
        neck_lyrs = []
        for _ in range(3):
            class_lyr = Conv2D(filters=num_classes + 1,
                                kernel_size=1,
                                padding="same",
                                activation="softmax")

            detection_lyr = Conv2D(filters=4,
                                   kernel_size=1,
                                   padding="same")

            neck_lyr = Concatenate(axis=-1)([detection_lyr(backbone_lyr), 
                                             class_lyr(backbone_lyr)])

            neck_lyr = Reshape([-1, 5 + num_classes])(neck_lyr)
            neck_lyrs.append(neck_lyr)
        out_lyr = Concatenate(axis=1)(neck_lyrs)
        out_layers.append(out_lyr)
    out = Concatenate(axis=1)(out_layers)
    return Model(inputs=backbone.inputs, outputs=out)

def get_neckbottle(backbone):
    #TODO
    pass

def get_model(**kwargs):
    tf.keras.backend.clear_session()
    backbone = get_backbone(kwargs["input_shape"], 
                            kwargs["backbone_name"])

    head = get_head(kwargs["num_classes"], backbone)
    return head


if __name__ == "__main__":
    backbone = get_backbone((256, 256, 3), backbone_name="mobilenetv2")
    model = get_model(input_shape = (256, 256, 3),
                      backbone_name = "mobilenetv2",
                      num_classes = 2)
    backbone.summary()
    model.summary()
    plot_model(model, show_layer_names=False)
