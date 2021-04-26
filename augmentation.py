import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copy2, move
from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('img_path', './DatasetE2/JPEGImages', 'path for input images')
flags.DEFINE_string('masks_path', './DatasetE2/SegmentationClass', 'path for label images')
flags.DEFINE_string('augmented_path', None, 'path for augmented dataset')
flags.DEFINE_string('labels','./DatasetE2/labelmap.txt','path for the labels description')
flags.DEFINE_integer('n_images',192,'number of images to generate')

# Data generator:
datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=180,
    width_shift_range=0.5,
    height_shift_range=0.5,
    zoom_range = 0.7,
    shear_range = 45,
    vertical_flip = True,
    horizontal_flip = True
    )

def main(argv_):

    # Initialize variables

    ImgDir = FLAGS.img_path
    MasksDir = FLAGS.masks_path
    results_path = FLAGS.augmented_path
    labels_path = FLAGS.labels
    n_images = FLAGS.n_images

    if not FLAGS.augmented_path:
        try:
            os.mkdir('./AugmentedDataset')
        except:
            pass
        results_path = './AugmentedDataset'
        

    try:
      os.mkdir(results_path+'/Images')
      os.mkdir(results_path+'/Masks')
    except:
      pass
  
    copy2(labels_path,results_path)
    
    f=1
    if not(len(os.listdir(ImgDir)) == 1 and os.listdir(ImgDir)[0].find('.') == -1):
        while f:
          try:
            os.mkdir(ImgDir+'/../Images'+str(f))
            j=f
            f=0
          except:
            f+=1
        ImgDir_2 = ImgDir[:ImgDir.rfind('/')]+'/Images'+str(j)
        move(ImgDir,ImgDir+'/../Images'+str(j))
    else:
        ImgDir_2 = ImgDir  

    f=1
    if not(len(os.listdir(MasksDir)) == 1 and os.listdir(MasksDir)[0].find('.') == -1):
        while f:
          try:
            os.mkdir(MasksDir+'/../Masks'+str(f))
            j=f
            f=0
          except:
            f+=1
        MasksDir_2 = MasksDir[:MasksDir.rfind('/')]+'/Masks'+str(j)
        move(MasksDir,MasksDir+'/../Masks'+str(j))
    else:
        MasksDir_2 = MasksDir

    
    image_size = plt.imread(MasksDir_2+'/'+os.listdir(MasksDir_2)[0]+'/'+next(os.walk(MasksDir_2+'/'+os.listdir(MasksDir_2)[0]))[2][0]).shape[:2]

    seed = np.random.randint(100)
    image_generator = datagen.flow_from_directory(directory=ImgDir_2,target_size=image_size,save_to_dir=results_path+'/Images',
                                                  class_mode=None,save_format='jpg',seed = seed)

    masks_generator = datagen.flow_from_directory(directory=MasksDir_2,target_size=image_size,save_to_dir=results_path+'/Masks',
                                                  class_mode=None,save_format='png',seed = seed)


    n_iter = int(n_images/32)
    if n_images % 32:
      n_iter += 1
    
    print(f"Generating {n_iter*32} images")
    for i in range(n_iter):
      image_generator.next()
      masks_generator.next()
      print(i+1,'/',n_iter, 'done')


    # Reorder the directories
    if not f:
      move(MasksDir_2+'/'+os.listdir(MasksDir_2)[0], MasksDir_2+'/..')
      move(ImgDir_2+'/'+os.listdir(ImgDir_2)[0],ImgDir_2+'/..')
      os.rmdir(ImgDir_2)
      os.rmdir(MasksDir_2)

if __name__=="__main__":
    app.run(main)

    



    
        

