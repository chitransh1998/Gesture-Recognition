# Hand Gesture Recognition
A Deep Learning based Hand Gesture Recognition Tool

## Introduction
Hand gestures are powerful human to human non-verbal form of communication conveying major part of information transfer in our everyday life.  
Hand gesture recognition is a process of understanding and classifying meaningful movements my human hands which can then be used to perform different tasks.  

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Hand_gestures_into.jpg?raw=true)

Here,an Artificial Neural Network is used to classify hand gestures using a self generated dataset and the model used is a Convolutional Neural network.

This model can be used in applications like Home Automation, Sign Language Interpretation, Gaming or unlocking smartphones etc. with requisite further additions to the pipeline.

## Vision v/s Hardware Methods

Earlier hardware based gesture recognition was more prevalent.User had to wear gloves helmet and other heavy apparatus which made the process difficult in real time environment.  
In contrast,vision based methods require only a camera,thus realizing a natural interaction between humans and computers without the use of any extra devices.  

## Dataset

The dataset for the project was collected manually using a webcam in the author’s personal computer.A code was written
to automate and speed up the process where in multiple images were taken and stored in the respective directories.  
The dataset consists of 2320 images each 200*200 pixels and three channel wide,divided into 2000 training and 320 test
images.  
Further,each class has 500 training and 80 test images and the classes of gesture which are as follows:
* Peace
* Punch
* Stop
* Thumbs_up    
Each image was first shot using a webcam and a binary mask was then applied to obtaining a binarized and thresholded
image.Binary mask has been discussed in the preprocessing section in detail.  
The training data was used to train the network weights and the test data to validate the training.  


![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

## Pre Processing
Images are pre-processed before classification. There are two modes of processing based on background noise levels viz. Binary Mode Processing and Skin Mask Mode Processing.

### Binary Mode Processing
In this mode the image is first converted into a grayscale using OpenCV command RGB2GRAY. Following which Gaussian Blurring is applied with a kernel of 5*5 to blur the noise in the image.This is followed by a adaptive gaussian thresholding.  
The adaptive feature allows the model to select a threshold as per the image and thus making the code more robust to noise.   
Finally,the thresholded image is input to the model for classification.This mode finds it use when you have an empty background without any other objects like a white wall or a whiteboard.

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

## Skin Mode Processing
In this mode the source image first converted from RGB to YCrCb space using the BGR2YCR_CB function in OpenCV and saved with a different name.YCbCr represents a family of color spaces used as part of the color image pipeline in video and digital photography systems.Here,Y represents the luminance component and Cb and Cr represent the blue-difference and red-difference chroma components.  
Luma component is the brightness of the color. That means the light intensity of the color. The human eye is more sensitive to this component.

After converting to YCrCb color space the regions with the skin tone are easily identified by using a suitable range of values for minimum and maximum parameters in the cv2.inRange() function.  
Then contours are detected in the regions with the skin tone and finally the detected contours are drawn on the original source image.It is also made sure that only those contours are drawn whose areas is greater than 1000 pixel square to avoid small patches and noises.  
The original source image is then converted from RGB to GRAYSCALE and then binarized using cv2.threshold function.  
Finally the binarized image is passed to the neural network for classification.  

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

## Building the Model

Keras is used for building,training and testing the model.   
The model consists of two broad divisions namely:
1. Feature Extraction
2. Classification

### Feature Extraction
Feature extraction is a dimensionality reduction process,where an initial set of raw variables is reduced to more manageable
groups or features for processing,while still accurately and completely describing the original dataset.When the input data
to an algorithm is too large to be processed and it is suspected to be redundant (eg the repetitiveness of images presented as pixels),then it can be transformed into a reduced set of features.The selected features are expected to contain the
relevant information from the input data,so that the desired task can be performed using this reduced representation of the complete initial data.

The network consists of two convolutional layers with a ReLU activation function..Max Pooling in two dimensions and dropout layers have also been added to reduce overfitting to the training fata.

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

### Classification
Image Classification is assigning pixels in image to categories or classes of interest.In order to achieve this,the relationship between the data and classes into which they are classified must be well understood.To achieve this,we feed the features extracted using the convolution layer into a fully connected or dense layer.

The dense layer gets a horizontal input from the feature extraction part.There are three dense layers with ReLu activation functions.Each layer if followed by a dropout layer to reduce overfitting.The output layer consists of four outputs one for each gesture and has sigmoid activation function to give probabilistic interpretation of the outputs.

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*L to R: Thumbs_up, Stop, Punch, Peace*

## Training the Model
After building the model,it is compiled where the backend automatically chooses the best way to represent the network for
training and making predictions to run on hardware.It uses efficient numerical libraries like Theano or tensorflow for the
same.Training means finding the best set of weights and biases to make a prediction for the problem.  

The categorical cross entropy function is used to accuracy and evaluating the model.  
Adam optimizer is used to search for different weights to make prediction for this problem.  

### Hyperparameters for the Optimizer
1. ***Alpha***. Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right downduring training. `Alpha=0.001`    
2. ***Beta1***. The exponential decay rate for the first moment estimates (e.g. 0.9). `Beta1 = 0.9`  
3. ***Beta2***. The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems). `Beta2 = 0.999`  
4. ***Epsilon***. Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8). `Epsilon = 10e-8`  
5. ***Number of epochs*** =10
6. ***Batch Size*** = 32

Three models were trained using the above specified criteria and making small changes.The model with the highest accuracy
and minimum loss was taken.  

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*Model training*

## Results
The training results of the proposed model were found as follows:  
* Training Loss: 0.1762  
* Training Accuracy: 99.49%  
* Test Loss: 0.4382  
* Test Accuracy: 91.87%

A high training accuracy was observed due to the high amount of data available and well-tuned structure of the neural network. The model was then tested on live images using a webcam and the results were found to be satisfactory.    

![image](https://github.com/chitransh1998/Gesture-Recognition/blob/main/Dataset.png?raw=true)
*Curves*
The training and the testing accuracy curves show a steady increase in accuracy as the number of epochs progresses.Similarly,there is consistent decrease the testing and training loss with the subsequent epoch.Hence,the model
behaves well suited.  
Previously,I was using simply images with clean background to recognize gestures.Now,with earlier discussed pre-processing techniques it was possible to recognize gestures even without clear backgrounds.  

## Next Steps
The high training accuracy shows that the model might have overfitted to the training data.Hence,some alternate methods
and possible changes to improve the model are as follows:

● Regularization: Regularization modifies the objective
function that we minimize by adding additional terms
that penalize large weights. In other words, we change
the objective function so that it becomes Error+λf(θ),
where f(θ) grows larger as the components of θ grow
larger and λ is the regularization strength (a
hyper-parameter for the learning algorithm).The most
common type of regularization is L2 regularization. It
can be implemented by augmenting the error function
with the squared magnitude of all weights in the
neural network. In other words, for every weight w in
the neural network, we add 1/2 λw^2 to the error
function. The L2 regularization has the intuitive
interpretation of heavily penalizing "peaky" weight
vectors and preferring diffuse weight vectors
● Early stopping : With early stopping, the choice of the
validation set is also important. The validation set
should be representative of all points in the training
set.When you use Bayesian regularization, it is
important to train the network until it reaches
convergence. The sum-squared error, the sum-squared
weights, and the effective number of parameters
should reach constant values when the network has
converged.
● Data Augmentation:There are some data augmentation
techniques such as scaling, translation, rotation,
flipping,resizing.Data augmentation can help reduce
the manual interventation required to developed
meaningful information and insight of business data,
