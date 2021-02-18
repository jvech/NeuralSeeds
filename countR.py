import pandas as pd
from PIL import Image
import numpy as np 
from skimage.measure import regionprops_table,label
from skimage import filters,morphology
from skimage.color import rgb2gray 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 

def get_labels(img):
    img = rgb2gray(img) if img.shape[2] == 3 else np.squeeze(img)
    threshold = filters.threshold_otsu(img)
    mask = img > threshold
    mask = morphology.remove_small_objects(mask, 25)
    mask = morphology.remove_small_holes(mask, 25)
    labels = label(mask)
    return labels

def count(img):
    labels= get_labels(img)
    props = regionprops_table(labels, properties=('bbox',))
    dataframe = pd.DataFrame(props)
    return dataframe

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image','Predicted Mask','True Mask']

    for i in range(len(display_list)):
        ax =plt.subplot(1, len(display_list), i+1)
        ax.title.set_text(title[i])
        ax.imshow(display_list[i])

        if title[i] == 'Predicted Mask':
            bboxes=np.array(count(display_list[i]).iloc[:,:4])
            ax.title.set_text(title[i]+' seeds:'+str(bboxes.shape[0]))
            for bbox in bboxes:
                xy = (bbox[1],bbox[0])
                width = bbox[3]-bbox[1]
                heigth = bbox[2]-bbox[0]
                patch = patches.Rectangle(xy,width,heigth,linewidth=1,
                                        facecolor='none', edgecolor='r')
                ax.add_patch(patch)
        ax.axis('off')
    plt.show()

    
if __name__=="__main__":
    image =np.asarray(Image.open('./seeds_data/SegmentationClass/000.png'))
    print(count(image))
    display([image,image,image])
