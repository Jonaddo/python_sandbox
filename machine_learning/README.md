# [IN CONSTRUCTION]
Upcoming topics include the following:
- How to deal with imbalanced datasets?
- How to deal with missing values?
- How to deal with outliers?
- How to detect outliers?
- How to deal with categorical variables?

# About this repository

Here you will find different ML exercises related to classification, clustering and regression.

## Classification
File: *heart_attack_knn.ipynb*

The first exercise is done on medical data in order to predict if a patient's heart is potentialy at risk or not using a supervised learning clustering algorithm called K-NN.  

In statistics, the k-nearest neighbors algorithm (k-NN) is a non-parametric classification method.  
It is used for classification and regression.  
  
In k-NN classification, the output is a class membership. An object is classified by a plurality vote
of its neighbors, with the object being assigned to the class most common among its k nearest neighbors
(k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of
that single nearest neighbor (source: [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)).  

The dataset can be downloaded on Kaggle [here](https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) and the related jupyter notebook is the file called *heart_attack_knn.ipynb*.

### Workflow
1. Load the dataset
2. Clean the dataset
3. Exploratory Data Analysis (EDA)
4. Feature selection
5. Normalize/Standardized continous variable
6. Model
7. Grid Search cross-validation for hyperparmeter tuning
8. Predict out-of-sample


## Feature space reduction
File: *feature_space_reduction_AE.ipynb*

In this second exercise, we visit the neural-network of type **Autoencoder** (AE), here, to reduce the feature space dimension.  
Indeed, after training the **AE**, we can use the first part of the neural-network *aka* the encoder to encode/compress the input dataset.  
This technical have multiple benefits, one of them can increase a classifiers performance.  

In order to showcase the concept, we use a credit card transaction dataset and a gradient boosting decision tree.  
The main libraries are `scikit-learn` and `tensorflow`.



