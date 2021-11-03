[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
# U-Net network to segment seeds

A neural network based in the ssd framework used to detect germinating seeds.

# Installation
```shell
$ git clone https://github.com/jvech/NeuralSeeds.git
$ pip install requirements.txt
```

# Usage

## Train

Use the following command in order to train the model:

```
Usage:
    train.py [options] <imgs> <annotations> 

Options:
    -h --help               Show this message
    -H --history            Show the history of model metrics
    --model <file>          Save the trained model [default: ./model.h5]
    --batch <int>           Batch size [default: 8]
    --epochs <int>          Number of epochs [default: 40]
    --val_split <float>     Rate of the validation data [default: 0.0]
    --backbone <name>       Select the detector backbone [default: mobilenetv2]
```
__Available backbones__: mobilenetv2, vgg16

## Inference
#TODO

### Examples
```shell
$ python train.py /imgs_folder /xml_ann/folder --epochs 10 --batch 4 --val_split 0.2
```

## References
* [Accurate machine learning-based germination detection, prediction and quality assessment of three grain crops](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-020-00699-x)
