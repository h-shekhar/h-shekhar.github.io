---
layout: post
title: Digit Recognition using ConvNets in Python with Keras
subtitle: CNN MNIST Scratch
type: computer-vision
comments: true
---

## Loading the MNIST dataset in Keras

```python
# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt

# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
```
![Plot](/img/2018/projects/com-vis/num.jpg){:class="img-responsive"}

## Baseline Model with Multi-Layer Perceptrons

```python
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
```
```
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
 - 8s - loss: 0.2796 - acc: 0.9205 - val_loss: 0.1391 - val_acc: 0.9582
Epoch 2/10
 - 9s - loss: 0.1105 - acc: 0.9677 - val_loss: 0.0896 - val_acc: 0.9725
Epoch 3/10
 - 8s - loss: 0.0704 - acc: 0.9803 - val_loss: 0.0799 - val_acc: 0.9760
Epoch 4/10
 - 8s - loss: 0.0494 - acc: 0.9858 - val_loss: 0.0735 - val_acc: 0.9789
Epoch 5/10
 - 8s - loss: 0.0361 - acc: 0.9897 - val_loss: 0.0665 - val_acc: 0.9800
Epoch 6/10
 - 9s - loss: 0.0260 - acc: 0.9929 - val_loss: 0.0643 - val_acc: 0.9807
Epoch 7/10
 - 8s - loss: 0.0195 - acc: 0.9956 - val_loss: 0.0592 - val_acc: 0.9825
Epoch 8/10
 - 9s - loss: 0.0125 - acc: 0.9977 - val_loss: 0.0615 - val_acc: 0.9821
Epoch 9/10
 - 8s - loss: 0.0099 - acc: 0.9982 - val_loss: 0.0586 - val_acc: 0.9816
Epoch 10/10
 - 8s - loss: 0.0075 - acc: 0.9987 - val_loss: 0.0595 - val_acc: 0.9813
Baseline Error: 1.87%
```
## Simple Convolutional Neural Network

```python
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
```
```
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
 - 66s - loss: 0.2344 - acc: 0.9332 - val_loss: 0.0811 - val_acc: 0.9749
Epoch 2/10
 - 66s - loss: 0.0720 - acc: 0.9782 - val_loss: 0.0435 - val_acc: 0.9850
Epoch 3/10
 - 68s - loss: 0.0510 - acc: 0.9844 - val_loss: 0.0399 - val_acc: 0.9873
Epoch 4/10
 - 67s - loss: 0.0401 - acc: 0.9874 - val_loss: 0.0382 - val_acc: 0.9881
Epoch 5/10
 - 67s - loss: 0.0331 - acc: 0.9897 - val_loss: 0.0339 - val_acc: 0.9885
Epoch 6/10
 - 69s - loss: 0.0269 - acc: 0.9917 - val_loss: 0.0304 - val_acc: 0.9898
Epoch 7/10
 - 75s - loss: 0.0230 - acc: 0.9925 - val_loss: 0.0326 - val_acc: 0.9884
Epoch 8/10
 - 73s - loss: 0.0185 - acc: 0.9943 - val_loss: 0.0280 - val_acc: 0.9896
Epoch 9/10
 - 70s - loss: 0.0162 - acc: 0.9950 - val_loss: 0.0315 - val_acc: 0.9900
Epoch 10/10
 - 69s - loss: 0.0147 - acc: 0.9955 - val_loss: 0.0307 - val_acc: 0.9904
CNN Error: 0.96%
```
## Larger Convolutional Neural Network

```python
# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
```
```
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 88s 1ms/step - loss: 0.3864 - acc: 0.8777 - val_loss: 0.0894 - val_acc: 0.9720
Epoch 2/10
60000/60000 [==============================] - 93s 2ms/step - loss: 0.0974 - acc: 0.9693 - val_loss: 0.0520 - val_acc: 0.9836
Epoch 3/10
60000/60000 [==============================] - 87s 1ms/step - loss: 0.0692 - acc: 0.9780 - val_loss: 0.0380 - val_acc: 0.9883
Epoch 4/10
60000/60000 [==============================] - 85s 1ms/step - loss: 0.0556 - acc: 0.9826 - val_loss: 0.0332 - val_acc: 0.9891
Epoch 5/10
60000/60000 [==============================] - 83s 1ms/step - loss: 0.0486 - acc: 0.9853 - val_loss: 0.0371 - val_acc: 0.9875
Epoch 6/10
60000/60000 [==============================] - 80s 1ms/step - loss: 0.0417 - acc: 0.9865 - val_loss: 0.0350 - val_acc: 0.9882
Epoch 7/10
60000/60000 [==============================] - 81s 1ms/step - loss: 0.0385 - acc: 0.9880 - val_loss: 0.0347 - val_acc: 0.9881
Epoch 8/10
60000/60000 [==============================] - 81s 1ms/step - loss: 0.0348 - acc: 0.9893 - val_loss: 0.0307 - val_acc: 0.9894
Epoch 9/10
60000/60000 [==============================] - 81s 1ms/step - loss: 0.0322 - acc: 0.9895 - val_loss: 0.0259 - val_acc: 0.9909
Epoch 10/10
60000/60000 [==============================] - 80s 1ms/step - loss: 0.0298 - acc: 0.9903 - val_loss: 0.0269 - val_acc: 0.9917
Large CNN Error: 0.83%
```


_In case if you found something useful to add to this article or you found a bug in the code or would like to improve some points mentioned, feel free to write it down in the comments. Hope you found something useful here._
{: .box-warning}
