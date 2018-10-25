---
layout: post
title: Dimentionality Reduction using MNIST data set
subtitle: PCA, t-SNE
tags: [machine learning, python]
---

## What is Dimensionality Reduction?
In machine learning, dimensionality reduction is the process of converting any data which is present in a higher dimension to somewhat lower dimension by preserving the underlying information concisely. We often apply a bunch of dimensionality reduction technique to better visualize our data. But that is not always the case. Sometimes we might value the performance over precision so we could reduce high n dimension data to n' dimension (n' <  n) data so as to perform manipulation faster.

### MNIST Data set
The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. Know more about [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

### Let's Dive In

Loading data sets.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d0 = pd.read_csv('./mnist_train.csv')

print(d0.shape)
print(d0.head(5)) #print first 5 row of d0

#save the labels into variable into l
l = d0['label']

#drop the label feature and store the pixel data in d
d = d0.drop("label", axis=1)
```
Display Dimension of data set
```python
print(d.shape)
print(l.shape)
```
Display and plot a number
```python
plt.figure(figsize=(7,7))

idx = 801

grid_data = d.iloc[idx].as_matrix().reshape(28,28) #reshape from 1D to 2D pixel array
plt.imshow(grid_data, interpolation = "none", cmap = "gray")
plt.show()

print(l[idx])
```
![Plot a Number](/img/2018/08/digit.png){:class="img-responsive"}

## Principal Component Analysis [(PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)
### 2-D Visualization using PCA

Steps to perform PCA manually:

1. Load the data-sets.
2. Perform pre-processing - Standardizing the data.(Mean=0, Variance=1).
3. Find Co-Variance Matrix which is S = A^T * A.
4. Find Eigen Values and corresponding Eigen Vector from the above co-var matrix (in this example we are taking top two eigen values)(2 is the number of dim of the new feature sub-space which is <=d.)
5. Taking top 2 eigen vector and multiplying with sample data so as to get new coordinate system.
6. Plot the new coordinates.

Display the shape of sample data set.
Pick first 15K data-points to work on for time-effeciency.
We can perform the same analysis on all of 42K data-points.
```python
labels = l.head(15000)
data = d.head(15000)
print("The shape of the sample data = ", data.shape) #15k data-points with 784(28*28) features
```

Data pre-processing : Standardizing the data.
Standardize features by removing the mean and scaling to unit variance
```python
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data) 
print(standardized_data.shape)
```
Find Co-variance Matrix which is A^T * A.
Given matrix A is 15kx784 dimension & upon computing A^T * A will end up being 784x784 dimension.
```python
sample_data = standardized_data
#matrix multiplication using Numpy
covar_matrix = np.matmul(sample_data.T , sample_data)
print("The Shape of the Co-Variance matrix is = ", covar_matrix.shape) #(Co-Var matrix is of dxd size)
```
Finding the top two eigen values and corresponding eigen vector for projecting onto a 2-Dim space.
```python
from scipy.linalg import eigh
# the parameter 'eigvals' is defined (low value to high value) 
# eigh function will return the eigen values in asending order
# this code generates only the top 2 (782 and 783) eigenvalues since indexing start from 0.
values, vectors = eigh(covar_matrix, eigvals=(782,783))

print("Shape of the eigen vectors = ", vectors.shape)

# converting the eigen vectors into (2,d) shape for easyness of further computations
vectors = vectors.T

print("Updated shape of eigen vectors = ",vectors.shape)
# here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector
# here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector
```
Projecting the original data sample on the plane formed by two principal eigen vectors by vector-vector multiplication.
```python
new_coordinates = np.matmul(vectors, sample_data.T)
#taking top 2 eigen vector and multiplying with sample data so as to get new coordinate system
print("Resultant new data points' shape", vectors.shape, "X", sample_data.T.shape, "=" , new_coordinates.shape)
#For every point we get 2-dim representation which is 2x15k
#New coordinate have 2 rows each correspoing to the each new dimensions or  principle coordinate and 15k points.
```
Appending label to the 2d projected data (simple vertical stacking)
```python
new_coordinates = np.vstack((new_coordinates, labels)).T

# Creating a new data frame for plotting the labelled points
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
print(dataframe.head())
```
Plotting the 2D data points with seaborn
```python
import seaborn as sn
sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
```
![PCA Plotting](/img/2018/08/pca1.png){:class="img-responsive"}

### PCA using Scikit-Learn

Initializing the PCA
```python
from sklearn import decomposition
pca = decomposition.PCA()
```
Configuring the parameters
The number of components = 2
```python
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)

#pca reduced will contain the 2d projects of simple data
print("shape of pca_reduced.shape =", pca_data.shape)
```
Attaching the label for each 2-d data point 
```python
pca_data = np.vstack((pca_data.T, labels)).T

# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sn.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
```
![PCA Plotting](/img/2018/08/pca2.png){:class="img-responsive"}

### PCA for Dimentionality Reduction (not for visualization)

```python
pca.n_components = 784
pca_data = pca.fit_transform(sample_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()

# If we take 200-dimensions, approx. 90% of variance is expalined.
```
![PCA Plotting](/img/2018/08/pca3.png){:class="img-responsive"}

## t-distributed stochastic neighbor embedding [(t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
### t-SNE using Scikit-Learn

```python
from sklearn.manifold import TSNE

#Picking the top 1000 points as TSNE takes a lot for 15k points
data_1000 = standardized_data[0:1000,:]
labels_1000 = labels[0:1000]

model = TSNE(n_components=2, random_state=0)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data_1000)

#Creating a new data frame which help us in plotting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim1", "Dim2", "label"))

#Plotting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim1', 'Dim2').add_legend()
plt.show()
```
![t-SNE Plotting](/img/2018/08/tsne.png){:class="img-responsive"}

### A variation of t-SNE with perplexity=50

```python
model = TSNE(n_components=2, random_state=0, perplexity=50)
tsne_data = model.fit_transform(data_1000) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 50')
plt.show()
```
![t-SNE Plotting](/img/2018/08/tsne1.png){:class="img-responsive"}

### A variation of t-SNE with perplexity=50 and #iteration=5000

```python
model = TSNE(n_components=2, random_state=0, perplexity=50,  n_iter=5000)
tsne_data = model.fit_transform(data_1000) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 50, n_iter=5000')
plt.show()
```
![t-SNE Plotting](/img/2018/08/tsne2.png){:class="img-responsive"}

### A variation of t-SNE with perplexity=2

```python
model = TSNE(n_components=2, random_state=0, perplexity=2)
tsne_data = model.fit_transform(data_1000) 

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 2')
plt.show()
```
![t-SNE Plotting](/img/2018/08/tsne3.png){:class="img-responsive"}


_In case if you found something useful to add to this article or you found a bug in the code or would like to improve some points mentioned, feel free to write it down in the comments. Hope you found something useful here._
{: .box-warning}
