import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Dataset
import tensorflow_addons as tfa
from PIL import Image
from skimage.filters import gaussian
import os
import shutil

from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('img_path', './DatasetE2/JPEGImages/', 'path for input images')
flags.DEFINE_string('masks_path', './DatasetE2/SegmentationClass/', 'path for label images')
flags.DEFINE_string('augmented_path', None, 'path for augmented dataset')
flags.DEFINE_string('labels','./DatasetE2/labelmap.txt','path for the labels description')


# Transformations:

def RandCrop(im, mask, seed=np.random.randint(0,255)):
  Height = im.shape[0]
  Width = im.shape[1]
  Crop = np.random.randint(500,Height)
  PaddingH = int((Height-Crop)/2)
  PaddingW = int((Width-Crop)/2)

  tf.random.set_seed(seed)
  img = tf.image.random_crop(im,[Crop, Crop, 3]).numpy()

  tf.random.set_seed(seed)
  img_mask = tf.image.random_crop(mask,[Crop, Crop, 3]).numpy()

  img = tf.image.pad_to_bounding_box(img,PaddingH,PaddingW,Height,Width).numpy()
  img_mask = tf.image.pad_to_bounding_box(img_mask,PaddingH,PaddingW,Height,Width).numpy()

  return img, img_mask

def RandFlipLR(im, mask, seed=np.random.randint(0,255)):
  tf.random.set_seed(seed)
  img = tf.image.random_flip_left_right(im).numpy()

  tf.random.set_seed(seed)
  img_mask = tf.image.random_flip_left_right(mask).numpy()
  return img, img_mask

def RandFlipUD(im, mask, seed=np.random.randint(0,255)):
  tf.random.set_seed(seed)
  img = tf.image.random_flip_up_down(im).numpy()

  tf.random.set_seed(seed)
  img_mask = tf.image.random_flip_up_down(mask).numpy()
  return img, img_mask

def RandRot(im, mask, seed=0):
  seed = np.random.random()*2*np.pi
  img = tfa.image.rotate(im,seed,fill_mode='nearest').numpy()

  img_mask = tfa.image.rotate(mask,seed,fill_mode='nearest').numpy()
  return img, img_mask

def RandZoomIn(im, mask, seed=np.random.randint(0,255)):
  Height = im.shape[0]
  Width = im.shape[1]
  Crop = np.random.randint(500,Height)

  tf.random.set_seed(seed)
  img = tf.image.random_crop(im,[Crop, Crop, 3]).numpy()

  tf.random.set_seed(seed)
  img_mask = tf.image.random_crop(mask,[Crop, Crop, 3]).numpy()

  img = tf.image.resize(img,(Height,Width),method='nearest').numpy()
  img_mask = tf.image.resize(img_mask,(Height,Width),method='nearest').numpy()
  return img, img_mask

def RandZoomOut(im, mask, seed=np.random.random()):
  Height = im.shape[0]
  Width = im.shape[1]
  ZoomH = np.random.randint(500,Height)
  ZoomW = np.random.randint(500,Width)
  PaddingH = int((Height-ZoomH)/2)
  PaddingW = int((Width-ZoomW)/2)

  img = tf.image.resize(im,(ZoomH, ZoomW),preserve_aspect_ratio=True,method='nearest').numpy()

  img_mask = tf.image.resize(mask,(ZoomH, ZoomW),preserve_aspect_ratio=True,method='nearest').numpy()

  img = tf.image.pad_to_bounding_box(img,PaddingH,PaddingW,Height,Width).numpy()
  img_mask = tf.image.pad_to_bounding_box(img_mask,PaddingH,PaddingW,Height,Width).numpy()
  return img, img_mask

def RandBright(im,mask,maxdelta=0.2):
  img = tf.image.random_brightness(im,maxdelta).numpy()
  return img,mask

def RandContr(im,mask,LoBound=0.55, UpBound=2.5):
  img = tf.image.random_contrast(im,LoBound,UpBound).numpy()
  return img, mask

def RandBlurr(im,mask,sigma=np.random.random()*6):
  img= (gaussian(im,sigma)*255).astype(np.uint8)
  return img, mask



def main(argv_):

    ImgDir = FLAGS.img_path
    MasksDir = FLAGS.masks_path
    results_path = FLAGS.augmented_path
    labels_path = FLAGS.labels

    if not FLAGS.augmented_path:
        try:
            os.mkdir('./AugmentedDataset')
        except:
            pass
        results_path = './AugmentedDataset'
    results_images = results_path+'/JPEGImages/' 
    results_masks = results_path+'/SegmentationClass/'
    
    try:
        os.mkdir(results_images)
        os.mkdir(results_masks)
    except:
        pass

    shutil.copy2(labels_path,results_path) # Copy labelmap.txt to the new created directory
    
    transformations = [RandFlipLR,RandBlurr,RandBright,RandContr,RandCrop,RandFlipUD,RandRot,RandZoomIn,RandZoomOut]
    h=0
    fig=plt.figure(figsize=(15,15))
    n_iter = 3
    for k in range(n_iter):
      for i in os.listdir(ImgDir):
        Im    = np.copy(Image.open(ImgDir+i))
        Mask  = np.copy(Image.open(MasksDir+i[0:-3]+'png'))

        func = transformations[np.random.randint(0,len(transformations))]
        
        Img=func(Im,Mask)
        if np.any(Img[1]) :
          Image.fromarray(Img[0]).save(results_images+str(h)+'.jpg')
          Image.fromarray(Img[1]).save(results_masks+str(h)+'.png')

        h+=1
      ImgDir = results_images
      MasksDir = results_masks
      print(h)


      

if __name__=="__main__":
    app.run(main)

    



    
        

