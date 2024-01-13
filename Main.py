#Main code to run the model using the trained weights


from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
from matplotlib.pyplot import imshow
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as K
a=1
# Constants for finding range of skin color in YCrCb
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)




#Function to generate binary mode masks
def binaryMask(frame):
    #Converting to grayscale
    frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY)
    #Gaussian blurring
    blur = cv2.GaussianBlur(frame,(5,5),2)
    #Adaptive thresholding
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th3

#Function to generate skin mode masking
def skinMask(sourceImage):
  # Convert image to YCrCb
  imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)
  # Find region with skin tone in YCrCb image
  skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
  # Do contour detection on skin region
  _,contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Draw the contour on the source image
  for i, c in enumerate(contours):
   area = cv2.contourArea(c)
   if area > 1000:
      #Drawing the contours
      cv2.drawContours(sourceImage, contours, i, (0,0,0),thickness=2)
  #Gray-scaling the source image  
  sourceImage = cv2.cvtColor( sourceImage, cv2.COLOR_RGB2GRAY)
  #Binarizing the source image
  _, res = cv2.threshold(sourceImage, 0, 255, cv2.THRESH_BINARY)
  return res

def load_model():
    try:
        #Loading the model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("weights.hdf5")
        print("Model successfully loaded from disk.")
        
        #compile the model again
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return model
    except:
        print("""Model not found""")
        return None
    



#realtime:
def realtime():
    #initialize preview window
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    
    if vc.isOpened(): #get the first frame
        rval, frame = vc.read()
        
    else:
        rval = False
    
    classes=["    PEACE  ","    PUNCH ","    STOP","    THUMBS-UP"]
    
    while rval:
        frame=cv2.flip(frame,1)
        cv2.rectangle(frame,(300,200),(500,400),(0,255,0),1)
        cv2.putText(frame,"Place your hand in the green box.", (50,50), cv2.FONT_HERSHEY_PLAIN , 1, 255)
        cv2.putText(frame,"Press esc to exit.", (50,100), cv2.FONT_HERSHEY_PLAIN , 1, 255)
        
        cv2.imshow("preview", frame)
        frame=frame[200:400,300:500]
        #frame = cv2.resize(frame, (200,200))
        
        if a==1:
          frame=binaryMask(frame)
          cv2.imshow("binary mask result", frame)
        else:
          frame=skinMask(frame)
          cv2.imshow("skin mask result", frame)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        if a==1:
          frame=frame.reshape((1,)+frame.shape)
          frame=frame.reshape(frame.shape+(1,))
        if a==2:
          frame=frame.reshape((1,)+frame.shape)
          frame=frame.reshape(frame.shape+(1,))

          
        m=test_datagen.flow(frame,batch_size=1)
        y_pred=model.predict_generator(m,1)
        print(classes[list(y_pred[0]).index(y_pred[0].max())])
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")
    vc=None
    

#loading the model

model=load_model()


if model is not None:
    
    print('Do you want binary mode processing or skin mode processing? Binary mode processing=1 , Skin mode processing=2')
    a = int(input("1 or 2 "))

    realtime()
    
