---
layout: post
title: Face Detection using OpenCV
subtitle: A gentle introduction 
type: computer-vision
comments: true
---

***Face detection*** is a computer technology being used in a variety of applications that identifies human faces in digital images.
Face detection can be regarded as a specific case of object detection.  
Face-detection algorithms focus on the detection of frontal human faces. It is analogous to image detection in which the image of a person is matched bit by bit. Image matches with the image stores in database. Any facial feature changes in the database will invalidate the matching process.

![Plot](/img/2018/projects/com-vis/face.png){:class="img-responsive"}

### Application
- Facial recognition
- Facial motion capture
- Biometrics
- Photography

### OpenCV Classifier
A computer program that decides whether an image is a positive image (face image) or negative image (non-face image) is called a classifier. A classifier is trained on hundreds of thousands of face and non-face images to learn how to classify a new image correctly. OpenCV provides us with two pre-trained and ready to be used for face detection classifiers:

- ***Haar Classifier***
- ***LBP Classifier***

Both of these classifiers process images in gray scales, basically because we don't need color information to decide if a picture has a face or not. As these are pre-trained in OpenCV, their learned knowledge files also come bundled with OpenCV opencv/data/.  

To run a classifier, we need to load the knowledge files first, as if it had no knowledge.
Each file starts with the name of the classifier it belongs to. For example, a Haar cascade classifier starts off as _haarcascade_frontalface_alt.xml_.  

These are the two types of classifiers we will be using to analyze image.

### HAAR Cascade Classifier

The Haar Classifier is a machine learning based approach, an algorithm proposed by Paul Viola and improved by Rainer Lienhart; which are trained from many positive images (with faces) and negatives images (without faces). It starts by extracting Haar features from each image as shown by the windows below:

![Plot](/img/2018/projects/com-vis/haarfeatures.png){:class="img-responsive"}

Each window is placed on the picture to calculate a single feature. This feature is a single value obtained by subtracting the sum of pixels under the white part of the window from the sum of the pixels under the black part of the window.  
Now, all possible sizes of each window are placed on all possible locations of each image to calculate plenty of features.

### LBP Cascade Classifier

As any other classifier, the Local Binary Patterns, or LBP in short, also needs to be trained on hundreds of images. Local binary patterns (LBP) is a type of visual descriptor used for classification in computer vision.  
It has since been found to be a powerful feature for texture classification; it has further been determined that when LBP is combined with the Histogram of oriented gradients (HOG) descriptor, it improves the detection performance considerably on some datasets.  
So, LBP features are extracted to form a feature vector that classifies a face from a non-face.  

***Concept***
- Each training image is divided into some blocks as shown in the picture below.
- For each block, LBP looks at 9 pixels (3×3 window) at a time, and with a particular interest in the pixel located in the center of the window.
- Then, it compares the central pixel value with every neighbor's pixel value under the 3×3 window. For each neighbor pixel that is greater than or equal to the center pixel, it sets its value to 1, and for the others, it sets them to 0.
- After that, it reads the updated pixel values (which can be either 0 or 1) in a clockwise order and forms a binary number. Next, it converts the binary number into a decimal number, and that decimal number is the new value of the center pixel. We do this for every pixel in a block.
- Then, it converts each block values into a histogram, so now we have gotten one histogram for each block in an image.
- Finally, it concatenates these block histograms to form a one feature vector for one image, which contains all the features we are interested. So, this is how we extract LBP features from a picture.

### Implementation

Importing required libraries.  
Importing time library for speed comparisons of both classifiers.  

```python
import cv2
import matplotlib.pyplot as plt
import time 
%matplotlib inline
```
When we load an image using OpenCV, it loads it into BGR color space by default. To show the colored image using matplotlib we have to convert it to RGB space. So, we need to define a function which does this operation.  

_cv2.cvtColor_ is an OpenCV function to convert images to different color spaces. It takes as input an image to transform, and a color space code (like cv2.COLOR_BGR2RGB) and returns the processed image.

```python
def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
Next, we define a function _detect_faces()_ which takes 3 arguments.

- ***f_cascade***: It defines the type of Cascade Classifier.
- ***colored_img***: It is the input image that we want to test.
- ***scaleFactor***: This function compensates a false perception in size that occurs when one face appears to be bigger than the other simply because it is closer to the camera.
***cv2.cvtColor*** is an OpenCV function to convert the test image to gray image as opencv face detector expects gray images.

_detectMultiScale(image, scaleFactor, minNeighbors)_: This is a general function to detect objects, in this case, it'll detect faces since we called in the face cascade. If it finds a face, it returns a list of positions of said face in the form “_Rect(x,y,w,h)_.”, if not, then returns “None”.

- ***image***: Converted input image is the grayscale image.
- ***minNeighbors***: This is a detection algorithm that uses a moving window to detect objects, it does so by defining how many objects are found near the current one before it can declare the face found. 

Once we have the list of recognized faces, we can loop over them and draw a rectangle on the copy of the image and return the modified copy of the picture.

```python
def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
 #just making a copy of image passed, so that passed image is not changed 
 img_copy = colored_img.copy()          
 
 #convert the test image to gray image as opencv face detector expects gray images
 gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
 
 #let's detect multiscale (some images may be closer to camera than others) images
 faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);          
 
 #go over list of faces and draw them as rectangles on original colored img
 for (x, y, w, h) in faces:
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)              
 
 return img_copy
 ```

### OpenCV HAAR Cascade Classifier

Here we will deal with detection. OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc. Those XML files are stored in the opencv/data/haarcascades/ folder.  
We need to load the required XML classifiers and then pass this as an argument in _detect_faces()_ function in order to test the new image.

```python
#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
```
### Test 1

```python
#load another image 
test2 = cv2.imread('data/a.jpg')  
 
#call our function to detect faces 
faces_detected_img = detect_faces(haar_face_cascade, test2)  
 
#convert image to RGB and show image 
plt.imshow(convertToRGB(faces_detected_img))
```
![Plot](/img/2018/projects/com-vis/t1.png){:class="img-responsive"}

### Test 2 - Dealing with False Positive

```python
#load another image 
test2 = cv2.imread('data/b.jpg')  
 
#call our function to detect faces 
faces_detected_img = detect_faces(haar_face_cascade, test2)  
 
#convert image to RGB and show image 
plt.imshow(convertToRGB(faces_detected_img))
```
![Plot](/img/2018/projects/com-vis/t2.png){:class="img-responsive"}

We got one false positives.

A simple tweak to the scale factor compensates for this so can move that parameter around. For example, scaleFactor=1.2 improved the results.

### Test 2 with scaleFactor=1.2

```python
#load another image
test2 = cv2.imread('data/b.jpg')  
 
#call our function to detect faces
faces_detected_img = detect_faces(haar_face_cascade, test2, scaleFactor=1.2)  
 
#convert image to RGB and show image
plt.imshow(convertToRGB(faces_detected_img))
```
![Plot](/img/2018/projects/com-vis/t3.png){:class="img-responsive"}

12 people detected

### OpenCV LBP Cascade Classifier

XML training files for LBP cascade are stored in the opencv/data/lbpcascades/ folder. We just need to pass this LBP classifier to the same function and then test new data.

```python
#load cascade classifier training file for lbpcascade 
lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')  
 
#load test image 
test2 = cv2.imread('data/a.jpg') 
 
#call our function to detect faces 
faces_detected_img = detect_faces(lbp_face_cascade, test2)  
 
#convert image to RGB and show image 
plt.imshow(convertToRGB(faces_detected_img))
```
![Plot](/img/2018/projects/com-vis/t1.png){:class="img-responsive"}

### Comparative analysis of HAAR vs. LBP 

Let's compare both Haar and LBP on two test images to see accuracy and time delay of each.

```python
#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml') 
#load cascade classifier training file for lbpcascade 
lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')  

#load test image2 
test = cv2.imread('data/c.jpg')
```
```python
#------------HAAR----------- 
#note time before detection 
t1 = time.time()  
 
#call our function to detect faces 
haar_detected_img = detect_faces(haar_face_cascade, test)  
 
#note time after detection 
t2 = time.time() 
#calculate time difference 
dt1 = t2 - t1 
#print the time difference

#------------LBP----------- 
#note time before detection 
t1 = time.time() 
 
#call our function to detect faces 
lbp_detected_img = detect_faces(lbp_face_cascade, test)  
 
#note time after detection 
t2 = time.time() 
#calculate time difference 
dt2 = t2 - t1 
#print the time difference

#create a figure of 2 plots (one for Haar and one for LBP) 
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  
 
#show Haar image 
ax1.set_title('Haar Detection time: ' + str(round(dt1, 3)) + ' secs') 
ax1.imshow(convertToRGB(haar_detected_img))  
 
#show LBP image 
ax2.set_title('LBP Detection time: ' + str(round(dt2, 3)) + ' secs') 
ax2.imshow(convertToRGB(lbp_detected_img))  
```
![Plot](/img/2018/projects/com-vis/t4.png){:class="img-responsive"}

### Observation

- ***Accuracy***: _HAAR_ detected more faces and than _LBP_.
- ***Speed***: _LBP_ was significantly faster than _HAAR_.