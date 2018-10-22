---
layout: post
title: How to use ConvNets in different ways
subtitle: CNN
tags: [deep learning, python, cnn]
category: Deep Learning
---
There are several ways we can use Convolutional Neural Network (CNN or ConvNets for short) for building a deep learning model. Some are computationally cheaper while some require deeper understanding of the underlying architecture. Each has their own advantages. Let's dive in to find out.

1. Building ConvNets from Scratch.
2. Use pre-trained ConvNets.
3. Use pre-trained ConvNets as fixed Feature Extractor.
4. Fine-tuning the Pre-trained ConvNets.

![Plot](/img/2018/10/cnn.jpg){:class="img-responsive"}

### 1. Building ConvNets from Scratch
To build a network from scratch, the architect is required to define the number of layers and filters, along with other tunable parameters. Training an accurate model from scratch also requires massive amounts of data, on the order of millions of samples, which can take an immense amount of time.

Creating a network from scratch means you determine the network configuration. This approach gives you the most control over the network and can produce impressive results, but it requires an understanding of the structure of a neural network and the many options for layer types and configuration.

While results can sometimes exceed transfer learning (see below), this method tends to require more images for training, as the new network needs many examples of the object to understand the variation of features. Training times are often longer, and there are so many combinations of network layers that it can be overwhelming to configure a network from scratch.

***General Steps:***

1. Build your own ConvNets from scratch.
2. Use dataset to train the model.
3. Classify the object.

(When to use: New dataset is large and very different from the original dataset)

### 2. Use pre-trained ConvNets - Transfer Learning
A common alternative to training a ConvNets from scratch is to use a pretrained model to automatically extract features from a new data set. This method, called transfer learning, is a convenient way to apply deep learning without a huge dataset and long computation and training time.

Transfer learning uses knowledge from one type of problem to solve similar problems. You start with a pretrained network and use it to learn a new task. One advantage of transfer learning is that the pretrained network has already learned a rich set of features. These features can be applied to a wide range of other similar tasks. For example, you can take a network trained on millions of images and retrain it for new object classification using only hundreds of images.

***General Steps:***

1. Take pre-trained ConvNets(VGGNet, AlexNet, etc).
2. Use similar kind of dataset to perform prediction as used by pre-trained ConvNets.
3. Use ConvNets directly(without modifying any parameter used in model) to classify the object.

### 3. Use pre-trained ConvNets as fixed Feature Extractor - Transfer Learning
Feature Extraction deals with extracting learned image features from a pretrained convolutional neural network, and use those features to train an image classifier. Feature extraction is the easiest and fastest way use the representational power of pretrained deep networks. For example, you can train a support vector machine, Logistic Regression and many more on the extracted features. Because feature extraction only requires a single pass through the data, it is a good starting point if you do not have a GPU to accelerate network training.

***General Steps:***

1. Take a ConvNets(like VGG16) pretrained on dataset(like ImageNet).
2. Remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet).
3. Then treat the rest of the ConvNets as a fixed feature extractor for the new dataset.
4. Use the extracted feature to train a linear classifier(e.g. Linear SVM or Softmax classifier) for the new dataset.
5. Classify the object.

### 4. Fine-tuning the Pre-trained ConvNets - Transfer Learning
Fine-tuning means taking weights of a trained neural network and use it as initialization for a new model being trained on data from the same domain. Fine-tuning a pretrained network with transfer learning is typically much faster and easier than training from scratch. It requires the least amount of data and computational resources. It is used to speed up the training and generally to overcome small dataset size.

There are various strategies, such as training the whole initialized network or "freezing" some of the pre-trained weights (usually whole layers).

***General Steps:***

1. Not only replace and retrain the classifier on top of the ConvNets on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation.
2. Fine-tune all the layers of the ConvNets, or keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. 

(When to use: New dataset is large and similar to the original dataset).

### How Companies Use ConvNets
![Plot](/img/2018/10/data.jpeg){:class="img-responsive"}

> “Data is a precious thing and will last longer than the systems themselves.” – [Tim Berners-Lee](https://en.wikipedia.org/wiki/Tim_Berners-Lee), inventor of the [World Wide Web](https://en.wikipedia.org/wiki/World_Wide_Web).

***DATA***. The companies that have lots of this magic four letter word are the ones that have an inherent advantage over the rest of the competition. The more training data that you can give to a network, the more training iterations you can make, the more weight updates you can make, and the better tuned to the network is when it goes to production. Facebook and Instagram can use all the photos of the billion users it currently has, Google can use search data, and Amazon can use data from the millions of products that are bought every day. And now you know the magic behind how they use it.

