#Code for obtaining augmented image shown in figure 17 of report

from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import numpy as np

#Setting the augment parameters to generate augmented image of the loaded image
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest')


img = load_img('punch489.jpg')  
x = img_to_array(img)  # creating a Numpy array with shape (3, 200, 200)
x = x.reshape((1,) + x.shape)  # converting to a Numpy array with shape (1, 3, 200, 200)

#Generating 500 augmented images and saving in respective directory.
i = 0
for batch in datagen.flow(x,save_to_dir='peace_new', save_prefix='punch', save_format='jpg'):
    i += 1
    if i > 500:
        break 