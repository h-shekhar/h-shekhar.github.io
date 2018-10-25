---
layout: post
title: Neural Artistic Style Transfer using Tensorflow
subtitle: Creating art with deeplearning
type: computer-vision
comments: true
---

## Introduction

In fine art, especially painting, humans have mastered the skill to create unique visual experiences through composing a complex interplay between the content and style of an image. Thus far the algorithmic basis of this process is unknown and there exists no artificial system with similar capabilities. However, in other key areas of visual perception such as object and face recognition near-human performance was recently demonstrated by a class of biologically inspired vision models called Deep Neural Networks. Here we introduce an artificial system based on a Deep Neural Network that creates artistic images of high perceptual quality. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images.

## Understanding the Objective

Neural style transfer is an optimization technique used where we have two images that we want to “blend” together. First is Content Image and second is Style Reference Image.
- ***Content Image*** is the image whose content we want to keep. 
- ***Style Reference Image*** is the image whose style we want to keep.

![Plot](/img/2018/projects/com-vis/tue.jpg){:class="img-responsive"}
_Fig: Content Image_  

![Plot](/img/2018/projects/com-vis/sn.jpg){:class="img-responsive"}
_Fig: Style Reference Image_  

Using these two image we need to create a base image such that it depict the _content_ of the Content Image and painted in the _style_ of the Style Image. This base image will then evolve as a final image as we keep minimizing the content and style loss functions.

## Define Content and Style representations

In order to get both the content and style representations of our image, we will look at some intermediate layers within our model. Intermediate layers represent feature maps that become increasingly higher ordered as you go deeper. In this case, we are using the network architecture VGG19, a pretrained image classification network. These intermediate layers are necessary to define the representation of content and style from our images. For an input image, we will try to match the corresponding style and content target representations at these intermediate layers. 

First we load VGG19, and feed in our input tensor to the model. This will allow us to extract the feature maps of the content, style, and generated images.
In a nutshell, we first feed this VGG model with our input tensor and pull out these intermediate layers from our network.

Then we need to define the layer of interest that we need to pull from the generated model.

```python
# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
```

We’ll load our pretrained image classification network. Then we grab the layers of interest as we defined earlier. Then we define a Model by setting the model’s inputs to an image and the outputs to the outputs of the style and content layers. In other words, we created a model that will take an input image and output the content and style intermediate layers.

## Define Loss Functions

We need to define two loss function ***Content Loss*** and ***Style Loss***.

## Content Loss

Content loss definition is actually quite simple. We’ll pass the network both the desired content image and our base input image. This will return the intermediate layer outputs from our model. Then we simply take the euclidean distance between the two intermediate representations of those images.

![Plot](/img/2018/projects/com-vis/cl.png){:class="img-responsive"}

Fˡᵢⱼ(x) and Pˡᵢⱼ(x) describe the respective intermediate feature representation of the network with inputs x and p at layer l .

We perform backpropagation in the usual way such that we minimize this content loss. We thus change the initial image until it generates a similar response in a certain layer as the original content image.

```python
def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))
```

## Style Loss

Computing style loss is a bit more involved, but follows the same principle, this time feeding our network the base input image and the style image. However, instead of comparing the raw intermediate outputs of the base input image and the style image, we instead compare the Gram matrices of the two outputs.

Mathematically, we describe the style loss of the base input image, x, and the style image, a, as the distance between the style representation (the gram matrices) of these images. We describe the style representation of an image as the correlation between different filter responses given by the Gram matrix Gˡ, where Gˡᵢⱼ is the inner product between the vectorized feature map i and j in layer l. We can see that Gˡᵢⱼ generated over the feature map for a given image represents the correlation between feature maps i and j.

To generate a style for our base input image, we perform gradient descent from the content image to transform it into an image that matches the style representation of the original image. We do so by minimizing the mean squared distance between the feature correlation map of the style image and the input image. The contribution of each layer to the total style loss is described by

![Plot](/img/2018/projects/com-vis/sl.png){:class="img-responsive"}

where Gˡᵢⱼ and Aˡᵢⱼ are the respective style representation in layer l of input image x and style image a. Nl describes the number of feature maps, each of size Ml=height∗width. Thus, the total style loss across each layer is

![Plot](/img/2018/projects/com-vis/sl1.png){:class="img-responsive"}

where we weight the contribution of each layer’s loss by some factor wl. 

```python
def gram_matrix(input_tensor):
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)
 
def get_style_loss(base_style, gram_target):
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  return tf.reduce_mean(tf.square(gram_style - gram_target))
```

## Run Gradient Descent

In order to optimize our loss function we can use the Adam optimizer or L-BFGS optimizer. We iteratively update our output image such that it minimizes our loss: we don’t update the weights associated with our network, but instead we train our input image to minimize loss. In order to do this, we must know how we calculate our loss and gradients.

### Compute Feature Representations

We’ll define a feature representation function that will load our content and style image, feed them forward through our network, which will then output the content and style feature representations from our model.

```python
def get_feature_representations(model, content_path, style_path):
  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img(style_path)
  
  # batch compute content and style features
  stack_images = np.concatenate([style_image, content_image], axis=0)
  model_outputs = model(stack_images)
  
  # Get the style and content feature representations from our model  
  style_features = [style_layer[0] for style_layer in model_outputs[:num_style_layers]]
  content_features = [content_layer[1] for content_layer in model_outputs[num_style_layers:]]
  return style_features, content_features
```
### Compute the loss.

```python
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
  
  style_weight, content_weight, total_variation_weight = loss_weights
  model_outputs = model(init_image)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  
  style_score = 0
  content_score = 0

  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
  style_score *= style_weight
  content_score *= content_weight
  total_variation_score = total_variation_weight * total_variation_loss(init_image)

  # Get total loss
  loss = style_score + content_score + total_variation_score 
  return loss, style_score, content_score, total_variation_score
```
### Compute the Gradients

We iteratively updated our image by applying our optimizers update rules using tf.gradient. The optimizer minimized the given losses with respect to our input image.

```python
def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss
```
### Define run style transfer process

```python
def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight = 1e-2): 
  display_num = 100
  
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  init_image = load_and_process_img(content_path)
  init_image = tfe.Variable(init_image, dtype=tf.float32)
 
  opt = tf.train.AdamOptimizer(learning_rate=10.0)
  iter_count = 1
  best_loss, best_img = float('inf'), None
  
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
```

## Run the model

To run it on the given image.

```python
best, best_loss = run_style_transfer(content_path, style_path, verbose=True, show_intermediates=True)
```
## Outcomes

Output after several iterations.

![Plot](/img/2018/projects/com-vis/st.png){:class="img-responsive"}

This comes the final output.

![Plot](/img/2018/projects/com-vis/out.png){:class="img-responsive"}

## References

- To run the code in google colab refer [here](https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb)
- https://arxiv.org/pdf/1508.06576.pdf

_In case if you found something useful to add to this article or you found a bug in the code or would like to improve some points mentioned, feel free to write it down in the comments. Hope you found something useful here._
{: .box-warning}
